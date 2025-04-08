import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init

# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
max_iters = 10000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 400
head_size = 16
n_embed = 128
n_head = 8
n_layer = 8
dropout = 0.1
num_experts = 8 # This can be adjusted depending on the overall number of parameters
top_k = 2 # This controls the number of active parameters
skip_prob_threshold = 0.5  # 用于训练时的跳过阈值

# 投机推理相关阈值（仅在推理时使用）
speculative_skip_threshold = 0.9   # 投机路由器预测的跳过概率阈值
speculative_verification_threshold = 0.1  # 投机验证允许的误差范围

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    data_local = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_local) - block_size, (batch_size,))
    x = torch.stack([data_local[i:i+block_size] for i in ix])
    y = torch.stack([data_local[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

# Expert module
class Expert(nn.Module):
    """ An MLP expert """
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

# 原有的跨层路由器
class CrossLayerRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(CrossLayerRouter, self).__init__()
        self.top_k = top_k
        self.expert_router = nn.Linear(n_embed, num_experts)
        self.noise_linear = nn.Linear(n_embed, num_experts)
        self.skip_router = nn.Linear(n_embed, 1)
    def forward(self, x):
        logits = self.expert_router(x)
        noise_logits = self.noise_linear(x)
        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise
        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        skip_logits = self.skip_router(x)
        skip_prob = torch.sigmoid(skip_logits)
        return router_output, indices, skip_prob

# 新增的投机推理相关模块：辅助快速路由器和门控验证器
class SpeculativeFastRouter(nn.Module):
    """ 轻量级的投机路由器，仅在推理时使用 """
    def __init__(self, n_embed):
        super().__init__()
        self.fc = nn.Linear(n_embed, 1)
    def forward(self, x):
        # x: [B, T, n_embed]
        logits = self.fc(x)  # [B, T, 1]
        return logits

class SpeculativeGating(nn.Module):
    """ 简单的门控验证器，用于验证投机跳过的合理性 """
    def __init__(self, n_embed):
        super().__init__()
        self.fc = nn.Linear(n_embed, 1)
    def forward(self, x):
        logits = self.fc(x)  # [B, T, 1]
        return logits

# 跨层稀疏混合专家模块，增加投机推理功能
class CrossLayerSparseMoE(nn.Module):
    def __init__(self, n_embed, num_experts, top_k, capacity_factor=1.0, use_speculative=False):
        super(CrossLayerSparseMoE, self).__init__()
        self.router = CrossLayerRouter(n_embed, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(n_embed) for _ in range(num_experts)])
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.num_experts = num_experts
        self.use_speculative = use_speculative
        # 如果采用投机推理，则初始化辅助模块
        if self.use_speculative:
            self.speculative_fast_router = SpeculativeFastRouter(n_embed)
            self.speculative_gating = SpeculativeGating(n_embed)
    
    def forward(self, x, return_skip_info=False):
        # x: [B, T, n_embed]
        batch_size, seq_len, _ = x.shape
        if (not self.training) and self.use_speculative:
            # 投机推理分支：使用辅助快速路由器判断跳过
            spec_fast_logits = self.speculative_fast_router(x)  # shape: [B, T, 1]
            spec_fast_prob = torch.sigmoid(spec_fast_logits).squeeze(-1)  # [B, T]
            spec_skip_decision = spec_fast_prob > speculative_skip_threshold  # 高概率跳过
            # 使用辅助门控验证器获得验证概率
            spec_gate_logits = self.speculative_gating(x)  # [B, T, 1]
            spec_gate_prob = torch.sigmoid(spec_gate_logits).squeeze(-1)  # [B, T]
            # 如果验证概率与快速预测的差异小于阈值，则认为投机有效
            verification_pass = (torch.abs(spec_gate_prob - spec_fast_prob) < speculative_verification_threshold)
            final_skip = spec_skip_decision & verification_pass  # [B, T]，为True则跳过当前层
            # 构造最终输出：对于跳过的token，直接复制输入；对于未跳过的token，调用原有专家计算（fallback）
            final_output = x.clone()
            non_skip_mask = ~final_skip
            if non_skip_mask.any():
                # 对未跳过的token，使用原有路由器计算结果
                gating_output, indices, _ = self.router(x)
                non_skip_x = x[non_skip_mask]
                non_skip_gating = gating_output[non_skip_mask]
                non_skip_indices = indices[non_skip_mask]
                flat_x = non_skip_x.view(-1, x.size(-1))
                flat_gating_output = non_skip_gating.view(-1, gating_output.size(-1))
                flat_indices = non_skip_indices.view(-1, indices.size(-1))
                tokens_per_batch = flat_x.size(0) * self.top_k
                expert_capacity = int((tokens_per_batch / self.num_experts) * self.capacity_factor)
                updates = torch.zeros_like(flat_x)
                for i, expert in enumerate(self.experts):
                    expert_mask = (flat_indices == i).any(dim=-1)
                    selected_indices = torch.nonzero(expert_mask).squeeze(-1)
                    if selected_indices.numel() > expert_capacity:
                        limited_indices = selected_indices[:expert_capacity]
                    else:
                        limited_indices = selected_indices
                    if limited_indices.numel() > 0:
                        expert_input = flat_x[limited_indices]
                        expert_output = expert(expert_input)
                        gating_scores = flat_gating_output[limited_indices, i].unsqueeze(1)
                        weighted_output = expert_output * gating_scores
                        updates.index_add_(0, limited_indices, weighted_output)
                non_skip_output = updates.view(non_skip_x.size())
                final_output[non_skip_mask] = non_skip_output
            if return_skip_info:
                return final_output, final_skip
            return final_output
        else:
            # 训练或不采用投机推理时：使用原有的路由器和专家计算
            gating_output, indices, skip_prob = self.router(x)
            skip_decision = (skip_prob > skip_prob_threshold).squeeze(-1)  # [B, T]
            final_output = x.clone()
            non_skip_mask = ~skip_decision
            if non_skip_mask.any():
                non_skip_x = x[non_skip_mask]
                non_skip_gating = gating_output[non_skip_mask]
                non_skip_indices = indices[non_skip_mask]
                flat_x = non_skip_x.view(-1, x.size(-1))
                flat_gating_output = non_skip_gating.view(-1, gating_output.size(-1))
                flat_indices = non_skip_indices.view(-1, indices.size(-1))
                tokens_per_batch = flat_x.size(0) * self.top_k
                expert_capacity = int((tokens_per_batch / self.num_experts) * self.capacity_factor)
                updates = torch.zeros_like(flat_x)
                for i, expert in enumerate(self.experts):
                    expert_mask = (flat_indices == i).any(dim=-1)
                    selected_indices = torch.nonzero(expert_mask).squeeze(-1)
                    if selected_indices.numel() > expert_capacity:
                        limited_indices = selected_indices[:expert_capacity]
                    else:
                        limited_indices = selected_indices
                    if limited_indices.numel() > 0:
                        expert_input = flat_x[limited_indices]
                        expert_output = expert(expert_input)
                        gating_scores = flat_gating_output[limited_indices, i].unsqueeze(1)
                        weighted_output = expert_output * gating_scores
                        updates.index_add_(0, limited_indices, weighted_output)
                non_skip_output = updates.view(non_skip_x.size())
                final_output[non_skip_mask] = non_skip_output
            if return_skip_info:
                return final_output, skip_decision
            return final_output

# 跨层Block
class CrossLayerBlock(nn.Module):
    """ 跨层Block: 包含自注意力层和MoE层 """
    def __init__(self, n_embed, n_head, num_experts, top_k, use_speculative=False):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.smoe = CrossLayerSparseMoE(n_embed, num_experts, top_k, use_speculative=use_speculative)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
    def forward(self, x, return_skip_info=False):
        x = x + self.sa(self.ln1(x))
        if return_skip_info:
            moe_output, skip_info = self.smoe(self.ln2(x), return_skip_info=True)
            x = x + moe_output
            return x, skip_info
        else:
            x = x + self.smoe(self.ln2(x))
            return x

# 跨层语言模型
class CrossLayerMoELanguageModel(nn.Module):
    def __init__(self, use_speculative=False):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.ModuleList([
            CrossLayerBlock(n_embed, n_head=n_head, num_experts=num_experts, top_k=top_k, use_speculative=use_speculative)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.register_buffer('layer_skip_counts', torch.zeros(n_layer))
        self.register_buffer('total_token_counts', torch.zeros(n_layer))
    def forward(self, idx, targets=None, collect_skip_stats=False):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        if collect_skip_stats:
            for i, block in enumerate(self.blocks):
                x, skip_info = block(x, return_skip_info=True)
                self.layer_skip_counts[i] += skip_info.sum().item()
                self.total_token_counts[i] += skip_info.numel()
        else:
            for block in self.blocks:
                x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    def get_skip_stats(self):
        if self.total_token_counts.sum() == 0:
            return torch.zeros_like(self.layer_skip_counts)
        return self.layer_skip_counts / self.total_token_counts
    def reset_skip_stats(self):
        self.layer_skip_counts.zero_()
        self.total_token_counts.zero_()
    def generate(self, idx, max_new_tokens, use_speculative=False):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

def kaiming_init_weights(m):
    if isinstance(m, (nn.Linear)):
        init.kaiming_normal_(m.weight)

def main():
    # 根据需要设置use_speculative为True，在推理时启用投机推理机制
    model = CrossLayerMoELanguageModel(use_speculative=True)
    model.apply(kaiming_init_weights)
    model = model.to(device)
    print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            model.reset_skip_stats()
            xb, yb = get_batch('train')
            _, _ = model(xb, yb, collect_skip_stats=True)
            skip_rates = model.get_skip_stats()
            losses = estimate_loss(model)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            print(f"Layer skip rates: {skip_rates}")
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
    
if __name__ == "__main__":
    main()
