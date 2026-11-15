
import torch
import math

# Three tokens, each with a 2-dimensional Q, K and V
Q = torch.tensor([
    [1.0, 1.0],  # cat
])

K = torch.tensor([
    [1.0, 0.0],  # The
    [0.0, 1.0],  # cat
    [1.0, 1.0],  # milk
])

V = torch.tensor([
    [10.0, 0.0],
    [0.0, 20.0],
    [10.0, 10.0],
])

# Step 1: Calculate similarity scores
scores = Q @ K.T

# Step 2: Scale the scores
head_dim = Q.shape[-1]
scaled_scores = scores / math.sqrt(head_dim)

# Step 3: Convert scores into attention weights
weights = torch.softmax(scaled_scores, dim=-1)

# Step 4: Collect information from Value vectors
output = weights @ V

print("Raw scores:")
print(scores)

print("\nScaled scores:")
print(scaled_scores)

print("\nAttention weights:")
print(weights)

print("\nAttention output:")
print(output)
