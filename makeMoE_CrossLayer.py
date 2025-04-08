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
skip_prob_threshold = 0.5  # 跳过当前层的概率阈值

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
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
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
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

#Multi-Headed Self Attention
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

#Expert module
class Expert(nn.Module):
    """ An MLP is a simple linear layer followed by a non-linearity i.e. each Expert """

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

# Cross-layer Router - Can decide whether to use current layer experts or skip to next layer
class CrossLayerRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(CrossLayerRouter, self).__init__()
        self.top_k = top_k
        # Expert selection router
        self.expert_router = nn.Linear(n_embed, num_experts)
        self.noise_linear = nn.Linear(n_embed, num_experts)
        
        # Skip decision network - Decides whether to skip current layer
        self.skip_router = nn.Linear(n_embed, 1)
        
    def forward(self, x):
        # Expert selection routing
        logits = self.expert_router(x)
        
        # Add noise
        noise_logits = self.noise_linear(x)
        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise
        
        # Select top-k experts
        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        
        # Skip decision - Output skip probability
        skip_logits = self.skip_router(x)
        skip_prob = torch.sigmoid(skip_logits)
        
        return router_output, indices, skip_prob

# Cross-layer Sparse Mixture of Experts Module
class CrossLayerSparseMoE(nn.Module):
    def __init__(self, n_embed, num_experts, top_k, capacity_factor=1.0):
        super(CrossLayerSparseMoE, self).__init__()
        self.router = CrossLayerRouter(n_embed, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(n_embed) for _ in range(num_experts)])
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.num_experts = num_experts
    
    def forward(self, x, return_skip_info=False):
        # Assuming x has shape [batch_size, seq_len, n_embd]
        batch_size, seq_len, _ = x.shape
        gating_output, indices, skip_prob = self.router(x)
        
        # Determine which inputs should skip the current layer
        # Skip if skip probability > threshold
        skip_decision = (skip_prob > skip_prob_threshold).squeeze(-1)  # [batch_size, seq_len]
        
        # Create output tensor, initialized as input (for skipped tokens, keep unchanged)
        final_output = x.clone()
        
        # Only process non-skipped inputs
        non_skip_mask = ~skip_decision
        if non_skip_mask.any():
            # Extract non-skipped inputs
            non_skip_x = x[non_skip_mask]
            non_skip_gating = gating_output[non_skip_mask]
            non_skip_indices = indices[non_skip_mask]
            
            # Process non-skipped inputs
            # Flatten processing
            flat_x = non_skip_x.view(-1, x.size(-1))
            flat_gating_output = non_skip_gating.view(-1, gating_output.size(-1))
            flat_indices = non_skip_indices.view(-1, indices.size(-1))
            
            tokens_per_batch = flat_x.size(0) * self.top_k
            expert_capacity = int((tokens_per_batch / self.num_experts) * self.capacity_factor)
            
            updates = torch.zeros_like(flat_x)
            
            for i, expert in enumerate(self.experts):
                expert_mask = (flat_indices == i).any(dim=-1)
                selected_indices = torch.nonzero(expert_mask).squeeze(-1)
                limited_indices = selected_indices[:expert_capacity] if selected_indices.numel() > expert_capacity else selected_indices
                
                if limited_indices.numel() > 0:
                    expert_input = flat_x[limited_indices]
                    expert_output = expert(expert_input)
                    gating_scores = flat_gating_output[limited_indices, i].unsqueeze(1)
                    weighted_output = expert_output * gating_scores
                    updates.index_add_(0, limited_indices, weighted_output)
            
            # Put processed results back into final output
            non_skip_output = updates.view(non_skip_x.size())
            final_output[non_skip_mask] = non_skip_output
        
        if return_skip_info:
            return final_output, skip_decision
        return final_output

# 跨层Block
class CrossLayerBlock(nn.Module):
    """ 跨层Block: 可以跳过当前层的专家计算 """

    def __init__(self, n_embed, n_head, num_experts, top_k):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.smoe = CrossLayerSparseMoE(n_embed, num_experts, top_k)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x, return_skip_info=False):
        # 注意力层总是执行
        x = x + self.sa(self.ln1(x))
        
        # MoE层可能被跳过
        if return_skip_info:
            moe_output, skip_info = self.smoe(self.ln2(x), return_skip_info=True)
            x = x + moe_output
            return x, skip_info
        else:
            x = x + self.smoe(self.ln2(x))
            return x

# 跨层语言模型
class CrossLayerMoELanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        
        # 使用ModuleList而不是Sequential，以便我们可以跟踪跳过信息
        self.blocks = nn.ModuleList([
            CrossLayerBlock(n_embed, n_head=n_head, num_experts=num_experts, top_k=top_k) 
            for _ in range(n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(n_embed) # final layer norm
        self.lm_head = nn.Linear(n_embed, vocab_size)
        
        # 用于跟踪跳过统计信息
        self.register_buffer('layer_skip_counts', torch.zeros(n_layer))
        self.register_buffer('total_token_counts', torch.zeros(n_layer))

    def forward(self, idx, targets=None, collect_skip_stats=False):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        
        # 处理每一层，可能跳过某些层的专家
        if collect_skip_stats:
            for i, block in enumerate(self.blocks):
                x, skip_info = block(x, return_skip_info=True)
                # 更新跳过统计信息
                self.layer_skip_counts[i] += skip_info.sum().item()
                self.total_token_counts[i] += skip_info.numel()
        else:
            for block in self.blocks:
                x = block(x)
        
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def get_skip_stats(self):
        """返回每层的跳过率统计信息"""
        if self.total_token_counts.sum() == 0:
            return torch.zeros_like(self.layer_skip_counts)
        return self.layer_skip_counts / self.total_token_counts

    def reset_skip_stats(self):
        """重置跳过统计信息"""
        self.layer_skip_counts.zero_()
        self.total_token_counts.zero_()

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

def kaiming_init_weights(m):
    if isinstance(m, (nn.Linear)):
        init.kaiming_normal_(m.weight)

def main():
    model = CrossLayerMoELanguageModel()
    model.apply(kaiming_init_weights)
    model = model.to(device)

    print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):
        # 每隔一段时间评估损失并收集跳过统计信息
        if iter % eval_interval == 0 or iter == max_iters - 1:
            # 重置跳过统计信息
            model.reset_skip_stats()
            
            # 收集一批数据的跳过统计信息
            xb, yb = get_batch('train')
            _, _ = model(xb, yb, collect_skip_stats=True)
            
            # 获取跳过率
            skip_rates = model.get_skip_stats()
            
            # 评估损失
            losses = estimate_loss(model)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            print(f"Layer skip rates: {skip_rates}")

        # 采样一批数据
        xb, yb = get_batch('train')

        # 评估损失
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    # 生成示例文本
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))

if __name__ == "__main__":
    main() 