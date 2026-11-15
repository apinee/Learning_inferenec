
import torch
import math

# Three tokens, each with a 2-dimensional vector
x = torch.tensor([
    [3.0, 4.0],
    [3.0, 4.0],
    [3.0, 4.0]
])

# Each token has a different position
positions = torch.tensor([0, 1, 2])

# For this tiny example, use one frequency
frequency = 1.0

# Calculate rotation angles
angles = positions * frequency

cos = torch.cos(angles)
sin = torch.sin(angles)

# Apply rotation to each token
x_rotated = torch.stack([
    x[:, 0] * cos - x[:, 1] * sin,
    x[:, 0] * sin + x[:, 1] * cos
], dim=-1)

print("Original vectors:")
print(x)

print("\nRotation angles:")
print(angles)

print("\nRotated vectors:")
print(x_rotated)
