import torch
import math

# Three tokens, each with 2 dimensions
Q = torch.tensor([
    [1.0, 1.0],
    [1.0, 0.0],
    [0.0, 1.0]
])

K = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
])

V = torch.tensor([
    [10.0, 0.0],
    [0.0, 20.0],
    [10.0, 10.0]
])

# 1. Calculate attention scores
scores = Q @ K.T

# 2. Scale scores
scores = scores / math.sqrt(Q.shape[-1])

# 3. Create causal mask
mask = torch.triu(
    torch.ones(3, 3),
    diagonal=1
).bool()

# 4. Block future tokens
scores = scores.masked_fill(mask, float("-inf"))

# 5. Softmax
weights = torch.softmax(scores, dim=-1)

# 6. Calculate attention output
output = weights @ V

print("Causal mask:")
print(mask)

print("\nMasked attention scores:")
print(scores)

print("\nAttention weights:")
print(weights)

print("\nAttention output:")
print(output)
