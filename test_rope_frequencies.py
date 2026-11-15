
import torch

# Simplified Qwen3 configuration
head_dim = 128
rope_theta = 1_000_000.0

# 64 dimension pairs
i = torch.arange(0, head_dim, 2).float()

# Calculate frequencies
frequencies = 1.0 / (
    rope_theta ** (i / head_dim)
)

# Three token positions
positions = torch.tensor([0, 1, 2]).float()

# Calculate position-dependent angles
angles = torch.outer(positions, frequencies)

print("Number of frequencies:", frequencies.shape)
print("Frequency tensor shape:", frequencies.shape)
print("Angle tensor shape:", angles.shape)

print("\nFirst 5 frequencies:")
print(frequencies[:5])

print("\nFirst 5 angles for each position:")
print(angles[:, :5])

# Apply RoPE to a sample vector
x = torch.randn(3, head_dim)

cos = torch.cos(angles)
sin = torch.sin(angles)

# Split vector into even and odd dimensions
x_even = x[:, 0::2]
x_odd = x[:, 1::2]

# Rotate each pair
rotated_even = x_even * cos - x_odd * sin
rotated_odd = x_even * sin + x_odd * cos

# Interleave even and odd dimensions
rotated = torch.stack(
    [rotated_even, rotated_odd], dim=-1
).flatten(-2)

print("\nOriginal shape:", x.shape)
print("Rotated shape:", rotated.shape)
print("RoPE implementation successful!")
