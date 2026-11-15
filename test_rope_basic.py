import torch
import math

# Original vector
x = torch.tensor([3.0, 4.0])

# Rotation angle
theta = math.pi / 2  # 90 degrees

# Calculate rotation
cos_theta = math.cos(theta)
sin_theta = math.sin(theta)

x_rotated = torch.tensor([
    x[0] * cos_theta - x[1] * sin_theta,
    x[0] * sin_theta + x[1] * cos_theta
])

print("Original vector:", x)
print("Rotated vector:", x_rotated)
