import torch
import math

torch.manual_seed(42)

# Small example: 3 tokens, hidden dimension 8
batch_size = 1
seq_len = 3
hidden_dim = 8
num_heads = 2
head_dim = hidden_dim // num_heads

# Input token representations
x = torch.randn(batch_size, seq_len, hidden_dim)

# Simplified projection layers
q_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
k_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
v_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
o_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)

# Create Q, K, V
Q = q_proj(x)
K = k_proj(x)
V = v_proj(x)

# Split into multiple heads
def split_heads(tensor):
    batch, seq, hidden = tensor.shape
    tensor = tensor.view(batch, seq, num_heads, head_dim)
    return tensor.transpose(1, 2)

Q = split_heads(Q)
K = split_heads(K)
V = split_heads(V)

print("Q shape:", Q.shape)
print("K shape:", K.shape)
print("V shape:", V.shape)

# Calculate attention
scores = Q @ K.transpose(-2, -1)
scores = scores / math.sqrt(head_dim)

# Causal mask
mask = torch.triu(
    torch.ones(seq_len, seq_len),
    diagonal=1
).bool()

scores = scores.masked_fill(mask, float("-inf"))

weights = torch.softmax(scores, dim=-1)

# Apply attention to values
output = weights @ V

print("\nAttention output before combining heads:")
print(output.shape)

# Combine heads
output = output.transpose(1, 2).contiguous()
output = output.view(batch_size, seq_len, hidden_dim)

print("\nOutput after combining heads:")
print(output.shape)

# Final output projection
output = o_proj(output)

print("\nFinal output shape:")
print(output.shape)
