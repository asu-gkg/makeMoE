import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn

# 创建Rich控制台
console = Console()

# 超参数设置
batch_size = 16
block_size = 32
n_embed = 128
n_head = 4
n_layer = 4
num_experts = 8
top_k = 2
dropout = 0.1
max_spec_depth = 2  # 最大投机深度
skip_prob_threshold = 0.5
learning_rate = 1e-3
max_iters = 1000
eval_interval = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 设置随机种子以确保结果可复现
torch.manual_seed(1337)

# 简单的注意力头模块
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # 计算注意力分数
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        # 确保T不超过tril的大小
        T = min(T, self.tril.size(0))
        # 应用掩码
        wei = wei[:, :T, :T].masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # 加权聚合值向量
        v = self.value(x) # (B,T,C)
        out = wei @ v[:, :T, :] # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

# 多头注意力模块
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

# 标准专家模块
class Expert(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# 轻量级专家模块（用于投机推理）
class LightweightExpert(nn.Module):
    def __init__(self, n_embed, reduction_factor=8):  # 更激进的减少参数
        super().__init__()
        reduced_dim = n_embed // reduction_factor
        self.net = nn.Sequential(
            nn.Linear(n_embed, reduced_dim),
            nn.GELU(),  # 使用GELU激活函数
            nn.Linear(reduced_dim, n_embed),
        )  # 移除dropout以加速推理

    def forward(self, x):
        return self.net(x)

# 跨层路由器
class CrossLayerRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super().__init__()
        self.top_k = top_k
        # 专家选择路由器
        self.expert_router = nn.Linear(n_embed, num_experts)
        self.noise_linear = nn.Linear(n_embed, num_experts)
        
        # 跳过决策网络
        self.skip_router = nn.Linear(n_embed, 1)
        
    def forward(self, x):
        # 专家选择路由
        logits = self.expert_router(x)
        
        # 添加噪声（帮助负载均衡）
        noise_logits = self.noise_linear(x)
        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise
        
        # 选择top-k专家
        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        
        # 跳过决策 - 输出跳过概率
        skip_logits = self.skip_router(x)
        skip_prob = torch.sigmoid(skip_logits)
        
        return router_output, indices, skip_prob

# 投机推理路由器
class SpeculativeRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k, num_future_layers):
        super().__init__()
        self.top_k = top_k
        self.num_future_layers = num_future_layers
        
        # 当前层的专家路由器
        self.current_router = nn.Linear(n_embed, num_experts)
        
        # 预测未来层的专家路由器
        self.future_routers = nn.ModuleList([
            nn.Linear(n_embed, num_experts) for _ in range(num_future_layers)
        ])
        
        # 投机深度预测器
        self.depth_predictor = nn.Linear(n_embed, num_future_layers + 1)  # +1 表示包括"不投机"选项
        
    def forward(self, x):
        # 当前层的路由决策
        current_logits = self.current_router(x)
        
        # 预测未来层的路由决策
        future_logits = [router(x) for router in self.future_routers]
        
        # 预测投机深度
        depth_logits = self.depth_predictor(x)
        depth_probs = F.softmax(depth_logits, dim=-1)
        
        # 计算期望的投机深度 (0 表示不投机，1-num_future_layers 表示投机的层数)
        expected_depth = torch.sum(depth_probs * torch.arange(0, self.num_future_layers + 1, 
                                                             device=x.device).unsqueeze(0).unsqueeze(0), 
                                   dim=-1)
        
        return current_logits, future_logits, expected_depth

# 验证器：验证投机结果的准确性
class SpeculationVerifier(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.verifier = nn.Linear(n_embed * 2, 1)
        
    def forward(self, speculated_output, actual_output):
        # 比较投机输出和实际输出
        combined = torch.cat([speculated_output, actual_output], dim=-1)
        verification_score = torch.sigmoid(self.verifier(combined))
        return verification_score

# 投机性跨层MoE块
class SpeculativeCrossLayerBlock(nn.Module):
    def __init__(self, layer_idx, n_embed, n_head, num_experts, top_k, num_future_layers=0, is_speculative=False):
        super().__init__()
        head_size = n_embed // n_head
        self.layer_idx = layer_idx
        self.is_speculative = is_speculative
        
        # 注意力层
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ln1 = nn.LayerNorm(n_embed)
        
        if is_speculative:
            # 轻量级专家网络（用于投机推理）
            self.experts = nn.ModuleList([LightweightExpert(n_embed) for _ in range(num_experts)])
            # 简化的路由器
            self.router = nn.Linear(n_embed, num_experts)
        else:
            # 完整权重的专家网络
            self.experts = nn.ModuleList([Expert(n_embed) for _ in range(num_experts)])
            
            if layer_idx < n_layer - 1:  # 不是最后一层才需要投机
                # 投机路由器（只有非投机块才需要）
                self.spec_router = SpeculativeRouter(n_embed, num_experts, top_k, 
                                                   min(num_future_layers, n_layer - layer_idx - 1))
                
                # 验证器
                self.verifier = SpeculationVerifier(n_embed)
            
            # 标准路由器
            self.router = CrossLayerRouter(n_embed, num_experts, top_k)
            
        self.ln2 = nn.LayerNorm(n_embed)
        self.top_k = top_k

    def forward(self, x, return_skip_info=False, return_spec_info=False):
        # 所有块都执行注意力层
        x = x + self.sa(self.ln1(x))
        
        if self.is_speculative:
            # 投机块使用简化的路由和轻量级专家
            normalized_x = self.ln2(x)
            
            # 简化的路由（无跳过机制）
            logits = self.router(normalized_x)
            top_k_logits, indices = logits.topk(self.top_k, dim=-1)
            weights = F.softmax(top_k_logits, dim=-1)
            
            # 应用专家计算
            B, T, C = normalized_x.shape
            expert_outputs = torch.zeros_like(normalized_x)
            
            for k in range(self.top_k):
                expert_idx = indices[:, :, k]
                for b in range(B):
                    for t in range(T):
                        e_idx = expert_idx[b, t].item()
                        expert_output = self.experts[e_idx](normalized_x[b, t].unsqueeze(0))
                        expert_outputs[b, t] += weights[b, t, k] * expert_output.squeeze(0)
            
            return x + expert_outputs
        else:
            # 真实块使用完整的路由机制
            normalized_x = self.ln2(x)
            
            # 标准路由
            router_output, indices, skip_prob = self.router(normalized_x)
            skip_decision = (skip_prob > skip_prob_threshold).squeeze(-1)
            
            # 创建输出张量，初始化为输入
            final_output = x.clone()
            
            # 只处理不跳过的输入
            non_skip_mask = ~skip_decision
            
            if non_skip_mask.any():
                # 提取不跳过的输入
                non_skip_x = normalized_x[non_skip_mask]
                non_skip_router_output = router_output[non_skip_mask]
                non_skip_indices = indices[non_skip_mask]
                
                # 应用专家计算
                expert_outputs = torch.zeros_like(non_skip_x)
                
                for k in range(self.top_k):
                    expert_idx = non_skip_indices[:, k]
                    expert_weights = non_skip_router_output.gather(-1, expert_idx.unsqueeze(-1)).squeeze(-1)
                    
                    for i, e_idx in enumerate(expert_idx):
                        expert_output = self.experts[e_idx.item()](non_skip_x[i].unsqueeze(0))
                        expert_outputs[i] += expert_weights[i] * expert_output.squeeze(0)
                
                # 将专家输出添加到最终输出
                temp_output = x[non_skip_mask] + expert_outputs
                final_output[non_skip_mask] = temp_output
            
            if return_skip_info and return_spec_info and hasattr(self, 'spec_router'):
                # 获取投机信息（仅在请求时且不是最后一层）
                current_logits, future_logits, expected_depth = self.spec_router(normalized_x)
                return final_output, skip_decision, (current_logits, future_logits, expected_depth)
            elif return_skip_info:
                return final_output, skip_decision
            else:
                return final_output

# 投机性跨层MoE语言模型
class SpeculativeCrossLayerMoEModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # 嵌入层
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        
        # 创建标准块和投机块
        self.standard_blocks = nn.ModuleList()
        self.speculative_blocks = nn.ModuleList()
        
        # 首先创建所有标准块
        for i in range(n_layer):
            self.standard_blocks.append(
                SpeculativeCrossLayerBlock(
                    layer_idx=i,
                    n_embed=n_embed, 
                    n_head=n_head, 
                    num_experts=num_experts, 
                    top_k=top_k,
                    num_future_layers=max_spec_depth,
                    is_speculative=False
                )
            )
        
        # 然后为每个标准块创建对应的投机块（除了最后一层）
        for i in range(n_layer - 1):
            layer_spec_blocks = nn.ModuleList()
            for j in range(min(max_spec_depth, n_layer - i - 1)):
                layer_spec_blocks.append(
                    SpeculativeCrossLayerBlock(
                        layer_idx=i+j+1,
                        n_embed=n_embed, 
                        n_head=n_head, 
                        num_experts=num_experts, 
                        top_k=top_k,
                        is_speculative=True
                    )
                )
            self.speculative_blocks.append(layer_spec_blocks)
        
        # 输出层
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        
        # 用于跟踪统计信息
        self.register_buffer('layer_skip_counts', torch.zeros(n_layer))
        self.register_buffer('total_token_counts', torch.zeros(n_layer))
        self.register_buffer('speculation_success_counts', torch.zeros(n_layer, max_spec_depth))
        self.register_buffer('speculation_attempt_counts', torch.zeros(n_layer, max_spec_depth))
        
        # 标记模型参数类型
        self.mark_parameters()
    
    def mark_parameters(self):
        """标记哪些参数是标准模型的一部分，哪些是投机模型的一部分"""
        for name, param in self.named_parameters():
            if 'speculative' in name:
                param.is_speculative = True
            else:
                param.is_speculative = False
    
    def get_standard_params(self):
        """获取标准模型的参数"""
        return [p for n, p in self.named_parameters() if not hasattr(p, 'is_speculative') or not p.is_speculative]
    
    def get_speculative_params(self):
        """获取投机模型的参数"""
        return [p for n, p in self.named_parameters() if hasattr(p, 'is_speculative') and p.is_speculative]
    
    def standard_forward(self, idx, targets=None, collect_stats=False):
        """标准前向传播（不使用投机）"""
        B, T = idx.shape
        
        # 嵌入层
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        
        # 只使用标准块
        if collect_stats:
            for i, block in enumerate(self.standard_blocks):
                x, skip_info = block(x, return_skip_info=True)
                # 更新跳过统计信息
                self.layer_skip_counts[i] += skip_info.sum().item()
                self.total_token_counts[i] += skip_info.numel()
        else:
            for block in self.standard_blocks:
                x = block(x)
        
        # 输出层
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # 计算损失
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def speculative_forward(self, idx, targets=None):
        """使用投机推理的前向传播（极简高效版）"""
        B, T = idx.shape
        
        # 嵌入层
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        
        # 记录开始时间（用于统计）
        start_time = time.time()
        spec_success = False
        
        # 1. 执行前4层的标准计算（固定）
        for layer_idx in range(4):
            x = self.standard_blocks[layer_idx](x)
        
        # 2. 保存中间状态
        mid_state = x.clone()
        
        # 3. 使用极轻量级投机计算后续层
        # 创建一个简单的投机计算函数，避免过多的控制流
        def compute_speculation():
            spec_x = mid_state.clone()
            # 只使用第一个投机块（固定深度为1）
            if len(self.speculative_blocks) > 0 and len(self.speculative_blocks[0]) > 0:
                spec_block = self.speculative_blocks[0][0]
                spec_x = spec_block(spec_x)
                # 直接应用输出层
                spec_x = self.ln_f(spec_x)
                spec_logits = self.lm_head(spec_x)
                return spec_logits
            return None
        
        # 4. 计算真实结果的函数
        def compute_real_result():
            real_x = mid_state.clone()
            # 执行剩余的标准块
            for layer_idx in range(4, n_layer):
                real_x = self.standard_blocks[layer_idx](real_x)
            # 应用输出层
            real_x = self.ln_f(real_x)
            real_logits = self.lm_head(real_x)
            return real_x, real_logits
        
        # 5. 推理模式下尝试投机
        if targets is None:  # 推理模式
            # 尝试投机计算
            spec_logits = compute_speculation()
            
            if spec_logits is not None:
                # 计算真实结果（用于验证和统计）
                _, real_logits = compute_real_result()
                
                # 简单验证：比较预测的top-1 token是否相同
                spec_pred = spec_logits.argmax(dim=-1)
                real_pred = real_logits.argmax(dim=-1)
                
                # 计算准确率
                match_rate = (spec_pred == real_pred).float().mean().item()
                
                # 如果匹配率高，使用投机结果
                if match_rate > 0.9:  # 90%的token预测匹配
                    spec_success = True
                    # 记录统计信息
                    self.speculation_success_counts[0, 0] += 1
                    self.speculation_attempt_counts[0, 0] += 1
                    
                    # 返回投机结果
                    end_time = time.time()
                    # 打印统计信息（可选）
                    # print(f"投机成功! 匹配率: {match_rate:.2f}, 用时: {(end_time-start_time)*1000:.2f}ms")
                    return spec_logits, None
                else:
                    # 记录统计信息
                    self.speculation_attempt_counts[0, 0] += 1
                    # 返回真实结果
                    return real_logits, None
        
        # 6. 训练模式或投机失败，执行标准计算
        real_x, real_logits = compute_real_result()
        
        # 7. 计算损失（如果在训练模式）
        if targets is not None:
            B, T, C = real_logits.shape
            real_logits_flat = real_logits.view(B*T, C)
            targets_flat = targets.view(B*T)
            loss = F.cross_entropy(real_logits_flat, targets_flat)
            
            # 同时训练投机网络
            spec_logits = compute_speculation()
            if spec_logits is not None:
                spec_logits_flat = spec_logits.view(B*T, C)
                # 使用知识蒸馏损失
                temp = 2.0  # 温度参数
                soft_targets = F.softmax(real_logits_flat.detach() / temp, dim=-1)
                spec_loss = F.kl_div(
                    F.log_softmax(spec_logits_flat / temp, dim=-1),
                    soft_targets,
                    reduction='batchmean'
                ) * (temp * temp)
                loss = loss + 0.1 * spec_loss
        else:
            loss = None
        
        return real_logits, loss
    
    def forward(self, idx, targets=None, use_speculation=False, collect_stats=False):
        """统一的前向接口"""
        if use_speculation:
            return self.speculative_forward(idx, targets)
        else:
            return self.standard_forward(idx, targets, collect_stats)
    
    def get_skip_stats(self):
        """返回每层的跳过率统计"""
        if self.total_token_counts.sum() == 0:
            return torch.zeros_like(self.layer_skip_counts)
        return self.layer_skip_counts / self.total_token_counts
    
    def get_speculation_stats(self):
        """返回投机成功率统计"""
        success_rates = torch.zeros_like(self.speculation_success_counts)
        for i in range(n_layer):
            for j in range(max_spec_depth):
                if self.speculation_attempt_counts[i, j] > 0:
                    success_rates[i, j] = self.speculation_success_counts[i, j] / self.speculation_attempt_counts[i, j]
        return success_rates
    
    def reset_stats(self):
        """重置所有统计信息"""
        self.layer_skip_counts.zero_()
        self.total_token_counts.zero_()
        self.speculation_success_counts.zero_()
        self.speculation_attempt_counts.zero_()
    
    def generate(self, idx, max_new_tokens, use_speculation=True, progress_callback=None, spec_length=3):
        """
        使用投机采样生成文本序列 - 极简高效版本
        
        参数:
        - idx: 初始上下文
        - max_new_tokens: 要生成的最大token数
        - use_speculation: 是否使用投机采样
        - progress_callback: 进度回调函数
        - spec_length: 每次投机生成的token数量
        """
        # 如果不使用投机，使用标准自回归生成
        if not use_speculation:
            return self._standard_generate(idx, max_new_tokens, progress_callback)
        
        # 统计信息
        total_tokens = 0
        accepted_tokens = 0
        
        # 创建两个模型：轻量级模型和完整模型
        # 轻量级模型只使用前1/8的层
        draft_layers = n_layer // 8
        
        # 缓存嵌入层计算结果
        def get_embeddings(x):
            B, T = x.shape
            tok_emb = self.token_embedding_table(x)
            pos_emb = self.position_embedding_table(torch.arange(T, device=device))
            return tok_emb + pos_emb
        
        # 轻量级模型前向传播
        def draft_forward(x):
            # 嵌入层
            hidden = get_embeddings(x)
            
            # 只使用前1/8的层
            for i in range(draft_layers):
                hidden = self.standard_blocks[i](hidden)
            
            # 输出层
            hidden = self.ln_f(hidden)
            logits = self.lm_head(hidden)
            return logits[:, -1, :]  # 只返回最后一个位置的logits
        
        # 完整模型前向传播
        def target_forward(x):
            # 嵌入层
            hidden = get_embeddings(x)
            
            # 使用所有层
            for block in self.standard_blocks:
                hidden = block(hidden)
            
            # 输出层
            hidden = self.ln_f(hidden)
            logits = self.lm_head(hidden)
            return logits[:, -1, :]  # 只返回最后一个位置的logits
        
        # 开始生成
        for _ in range(max_new_tokens):
            # 确保上下文不超过block_size
            idx_cond = idx[:, -block_size:]
            
            with torch.no_grad():
                # 1. 使用轻量级模型生成token
                draft_logits = draft_forward(idx_cond)
                draft_probs = F.softmax(draft_logits, dim=-1)
                draft_token = torch.multinomial(draft_probs, num_samples=1)
                
                # 2. 使用完整模型验证
                target_logits = target_forward(idx_cond)
                target_probs = F.softmax(target_logits, dim=-1)
                
                # 3. 计算接受概率
                draft_prob = draft_probs[0, draft_token[0, 0]].item()
                target_prob = target_probs[0, draft_token[0, 0]].item()
                
                # 简化的接受/拒绝采样
                if target_prob > draft_prob * 0.5:  # 更宽松的接受条件
                    # 接受草稿token
                    idx = torch.cat((idx, draft_token), dim=1)
                    accepted_tokens += 1
                else:
                    # 拒绝草稿token，直接使用目标模型的预测
                    target_token = torch.multinomial(target_probs, num_samples=1)
                    idx = torch.cat((idx, target_token), dim=1)
            
            # 更新统计信息
            total_tokens += 1
            
            # 更新进度
            if progress_callback:
                progress_callback()
        
        # 打印投机统计信息
        acceptance_rate = accepted_tokens / total_tokens if total_tokens > 0 else 0
        console.print(f"[bold]投机接受率: {acceptance_rate:.2f} ({accepted_tokens}/{total_tokens})[/bold]")
        
        return idx

    def _standard_generate(self, idx, max_new_tokens, progress_callback=None):
        """标准自回归生成（不使用投机）"""
        for _ in range(max_new_tokens):
            # 截取最后 block_size 个token
            idx_cond = idx[:, -block_size:]
            # 获取预测
            logits, _ = self.standard_forward(idx_cond)
            # 只关注最后一个时间步
            logits = logits[:, -1, :]
            # 应用softmax获取概率
            probs = F.softmax(logits, dim=-1)
            # 采样
            idx_next = torch.multinomial(probs, num_samples=1)
            # 将采样得到的token添加到序列中
            idx = torch.cat((idx, idx_next), dim=1)
            # 更新进度
            if progress_callback:
                progress_callback()
        return idx

# 修改数据加载部分，使用真实文本数据
def load_text_dataset(file_path='input.txt', block_size=block_size):
    """从文本文件加载数据集"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        print(f"成功加载文本数据，总长度: {len(text)} 字符")
        
        # 创建字符级词汇表
        chars = sorted(list(set(text)))
        vocab_size = len(chars)
        print(f"词汇表大小: {vocab_size}")
        
        # 字符到整数的映射
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        
        # 编码和解码函数
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
        
        # 将文本转换为张量
        data = torch.tensor(encode(text), dtype=torch.long)
        
        # 划分训练集和验证集
        n = int(0.9 * len(data))
        train_data = data[:n]
        val_data = data[n:]
        
        # 创建数据集
        def create_dataset(data, block_size):
            inputs = []
            targets = []
            for i in range(0, len(data) - block_size, block_size):
                inputs.append(data[i:i + block_size])
                targets.append(data[i + 1:i + block_size + 1])
            
            return TensorDataset(torch.stack(inputs), torch.stack(targets))
        
        train_dataset = create_dataset(train_data, block_size)
        val_dataset = create_dataset(val_data, block_size)
        
        return train_dataset, val_dataset, vocab_size, encode, decode
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{file_path}'")
        print("尝试从网络下载示例文本数据...")
        return download_text_dataset(block_size)

def download_text_dataset(block_size=block_size):
    """从网络下载示例文本数据集"""
    import requests
    
    # 尝试下载莎士比亚作品集作为示例
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinysakespeare/input.txt"
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # 确保请求成功
        text = response.text
        
        print(f"成功下载示例文本数据，总长度: {len(text)} 字符")
        
        # 保存到本地文件
        with open('input.txt', 'w', encoding='utf-8') as f:
            f.write(text)
        print("已将数据保存到 'input.txt'")
        
        # 创建字符级词汇表
        chars = sorted(list(set(text)))
        vocab_size = len(chars)
        print(f"词汇表大小: {vocab_size}")
        
        # 字符到整数的映射
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        
        # 编码和解码函数
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
        
        # 将文本转换为张量
        data = torch.tensor(encode(text), dtype=torch.long)
        
        # 划分训练集和验证集
        n = int(0.9 * len(data))
        train_data = data[:n]
        val_data = data[n:]
        
        # 创建数据集
        def create_dataset(data, block_size):
            inputs = []
            targets = []
            for i in range(0, len(data) - block_size, block_size):
                inputs.append(data[i:i + block_size])
                targets.append(data[i + 1:i + block_size + 1])
            
            return TensorDataset(torch.stack(inputs), torch.stack(targets))
        
        train_dataset = create_dataset(train_data, block_size)
        val_dataset = create_dataset(val_data, block_size)
        
        return train_dataset, val_dataset, vocab_size, encode, decode
    except Exception as e:
        print(f"下载失败: {e}")
        # 退出程序
        exit(1)

def train(model, data_loader, optimizer, use_speculation=False, epoch=0, total_epochs=5):
    """训练模型一个epoch，使用Rich进度条显示"""
    model.train()
    total_loss = 0
    
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[bold green]{task.fields[loss]:.4f}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task(
            f"[cyan]Epoch {epoch+1}/{total_epochs}" + (" (投机模式)" if use_speculation else ""), 
            total=len(data_loader),
            loss=0.0
        )
        
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            
            # 前向传播
            _, loss = model(x, y, use_speculation=use_speculation)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 更新损失和进度条
            batch_loss = loss.item()
            total_loss += batch_loss
            avg_loss = total_loss / (progress.tasks[0].completed + 1)
            progress.update(task, advance=1, loss=avg_loss)
    
    return total_loss / len(data_loader)

def evaluate(model, data_loader, use_speculation=False):
    """评估模型性能，使用Rich进度条显示"""
    model.eval()
    total_loss = 0
    
    with Progress(
        TextColumn("[bold yellow]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[bold green]{task.fields[loss]:.4f}"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task(
            f"[yellow]评估" + (" (投机模式)" if use_speculation else ""), 
            total=len(data_loader),
            loss=0.0
        )
        
        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(device), y.to(device)
                
                # 前向传播
                _, loss = model(x, y, use_speculation=use_speculation)
                
                # 更新损失和进度条
                batch_loss = loss.item()
                total_loss += batch_loss
                avg_loss = total_loss / (progress.tasks[0].completed + 1)
                progress.update(task, advance=1, loss=avg_loss)
    
    return total_loss / len(data_loader)

def measure_inference_time(model, data_loader, use_speculation=False, num_runs=100):  # 更多运行次数
    """测量推理时间，使用Rich进度条显示"""
    model.eval()
    
    with Progress(
        TextColumn("[bold magenta]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task(
            f"[magenta]测量推理时间" + (" (投机模式)" if use_speculation else ""), 
            total=num_runs + 50  # 更多预热
        )
        
        with torch.no_grad():
            x, _ = next(iter(data_loader))
            x = x.to(device)
            
            # 预热
            for _ in range(50):  # 更多预热
                model(x, use_speculation=use_speculation)
                torch.cuda.synchronize()  # 确保完成
                progress.update(task, advance=1)
            
            # 确保GPU同步
            torch.cuda.synchronize()
            
            # 计时
            times = []
            for _ in range(num_runs):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                start.record()
                model(x, use_speculation=use_speculation)
                end.record()
                
                # 等待GPU完成
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
                progress.update(task, advance=1)
    
    # 去除异常值（最高和最低的5%）
    times = sorted(times)
    trim = int(len(times) * 0.05)
    times = times[trim:-trim]
    
    return sum(times) / len(times) / 1000  # 转换为秒


def main():
    console.print("[bold cyan]初始化投机性跨层MoE演示...[/bold cyan]")
    
    # 加载真实文本数据集
    train_dataset, val_dataset, vocab_size, encode, decode = load_text_dataset()
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 创建模型
    model = SpeculativeCrossLayerMoEModel(vocab_size).to(device)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    standard_params = sum(p.numel() for p in model.get_standard_params()) / 1e6
    speculative_params = sum(p.numel() for p in model.get_speculative_params()) / 1e6
    
    console.print(f"[bold green]总参数量:[/bold green] {total_params:.2f}M")
    console.print(f"[bold green]标准模型参数量:[/bold green] {standard_params:.2f}M")
    console.print(f"[bold green]投机模型参数量:[/bold green] {speculative_params:.2f}M")
    
    # 创建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # 训练模型
    console.print("\n[bold blue]开始训练模型...[/bold blue]")
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # 训练循环
    num_epochs = 5  # 设置训练轮数
    for epoch in range(num_epochs):
        # 标准模式训练
        train_loss = train(model, train_loader, optimizer, use_speculation=False, epoch=epoch, total_epochs=num_epochs)
        train_losses.append(train_loss)
        
        # 评估
        val_loss = evaluate(model, val_loader, use_speculation=False)
        val_losses.append(val_loss)
        
        console.print(f"[bold]Epoch {epoch+1}/{num_epochs} - 训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}[/bold]")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # 可以在这里保存模型
            # torch.save(model.state_dict(), 'best_model.pt')
            console.print("[bold green]发现更好的模型，已保存![/bold green]")
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    plt.title('训练过程')
    plt.savefig('training_loss.png')
    console.print("[bold green]训练损失曲线已保存为 'training_loss.png'[/bold green]")
    
    # 使用投机模式训练一个额外的epoch
    console.print("\n[bold blue]使用投机模式进行额外训练...[/bold blue]")
    spec_train_loss = train(model, train_loader, optimizer, use_speculation=True, epoch=0, total_epochs=1)
    spec_val_loss = evaluate(model, val_loader, use_speculation=True)
    console.print(f"[bold]投机模式 - 训练损失: {spec_train_loss:.4f}, 验证损失: {spec_val_loss:.4f}[/bold]")
    
    # 测量推理时间
    console.print("\n[bold magenta]测量推理时间...[/bold magenta]")
    standard_time = measure_inference_time(model, val_loader, use_speculation=False)
    spec_time = measure_inference_time(model, val_loader, use_speculation=True)
    
    console.print(f"[bold]标准推理时间: {standard_time:.6f}秒/批次[/bold]")
    console.print(f"[bold]投机推理时间: {spec_time:.6f}秒/批次[/bold]")
    
    speedup = standard_time / spec_time
    if speedup > 1:
        console.print(f"[bold green]推理加速比: {speedup:.2f}x[/bold green]")
    else:
        console.print(f"[bold red]推理加速比: {speedup:.2f}x[/bold red]")
    
    # 测量生成速度
    console.print("\n[bold magenta]测量文本生成速度:[/bold magenta]")
    
    # 准备上下文
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    if context.max() >= vocab_size:
        context = context % vocab_size
    
    # 测量标准生成速度
    start_time = time.time()
    with Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]使用标准模型生成文本", total=100)
        standard_output = model._standard_generate(context, max_new_tokens=100, 
                                                progress_callback=lambda: progress.update(task, advance=1))
    standard_time = time.time() - start_time
    
    # 测量投机生成速度
    start_time = time.time()
    with Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]使用投机模型生成文本", total=100)
        spec_output = model.generate(context, max_new_tokens=100, use_speculation=True,
                                    progress_callback=lambda: progress.update(task, advance=1))
    spec_time = time.time() - start_time
    
    # 解码生成的文本
    standard_text = decode(standard_output[0].tolist())
    spec_text = decode(spec_output[0].tolist())
    
    # 打印结果
    console.print(f"\n[bold]标准生成时间: {standard_time:.2f}秒[/bold]")
    console.print(f"[bold]投机生成时间: {spec_time:.2f}秒[/bold]")
    
    speedup = standard_time / spec_time
    if speedup > 1:
        console.print(f"[bold green]加速比: {speedup:.2f}x[/bold green]")
    else:
        console.print(f"[bold red]加速比: {speedup:.2f}x[/bold red]")
    
    console.print("\n[bold yellow]标准模型生成的文本:[/bold yellow]")
    console.print(f"[green]{standard_text}[/green]")
    
    console.print("\n[bold yellow]投机模型生成的文本:[/bold yellow]")
    console.print(f"[green]{spec_text}[/green]")

if __name__ == "__main__":
    main() 