
import torch
import math

torch.manual_seed(42)

# Configuration
batch_size = 1
seq_len = 3
hidden_dim = 8
num_q_heads = 4
num_kv_heads = 2
head_dim = hidden_dim // num_q_heads

# Input
x = torch.randn(batch_size, seq_len, hidden_dim)

# Projection layers
q_proj = torch.nn.Linear(
    hidden_dim, num_q_heads * head_dim, bias=False
)
k_proj = torch.nn.Linear(
    hidden_dim, num_kv_heads * head_dim, bias=False
)
v_proj = torch.nn.Linear(
    hidden_dim, num_kv_heads * head_dim, bias=False
)
o_proj = torch.nn.Linear(
    num_q_heads * head_dim, hidden_dim, bias=False
)

# Project Q, K, V
Q = q_proj(x)
K = k_proj(x)
V = v_proj(x)

# Split into heads
def split_heads(tensor, num_heads):
    batch, seq, _ = tensor.shape
    tensor = tensor.view(
        batch, seq, num_heads, head_dim
    )
    return tensor.transpose(1, 2)

Q = split_heads(Q, num_q_heads)
K = split_heads(K, num_kv_heads)
V = split_heads(V, num_kv_heads)

print("Q:", Q.shape)
print("K:", K.shape)
print("V:", V.shape)

# Repeat K and V heads to match Q heads
groups = num_q_heads // num_kv_heads

K = K.repeat_interleave(groups, dim=1)
V = V.repeat_interleave(groups, dim=1)

print("\nK after sharing:", K.shape)
print("V after sharing:", V.shape)

# Attention
scores = Q @ K.transpose(-2, -1)
scores = scores / math.sqrt(head_dim)

# Causal mask
mask = torch.triu(
    torch.ones(seq_len, seq_len),
    diagonal=1
).bool()

scores = scores.masked_fill(mask, float("-inf"))

weights = torch.softmax(scores, dim=-1)
output = weights @ V

# Combine heads
output = output.transpose(1, 2).contiguous()
output = output.view(
    batch_size, seq_len, num_q_heads * head_dim
)

# Output projection
output = o_proj(output)

print("\nAttention output:", output.shape)
print("GQA implementation successful!")