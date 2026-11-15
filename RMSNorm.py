import torch
from transformers import AutoModelForCausalLM

model_id = "Qwen/Qwen2.5-1.5B"

hf_model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float32, 
    device_map="cpu",
    trust_remote_code=False 
)
hf_model.eval()

hf_norm = hf_model.model.layers[0].input_layernorm
dummy_input = torch.randn(1, 4, hf_norm.weight.shape[0])

with torch.no_grad():
    expected_output = hf_norm(dummy_input)

class CustomRMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Calculate mean of squares
        variance = x.pow(2).mean(-1, keepdim=True)
        # 2. Multiply by reciprocal square root: x * (1 / sqrt(var + eps))
        normed = x * torch.rsqrt(variance + self.eps)
        # 3. Scale by learnable weight
        return normed * self.weight

# Validation test:
custom_norm = CustomRMSNorm(hf_norm.weight.shape[0], eps=hf_norm.variance_epsilon)
custom_norm.weight.data.copy_(hf_norm.weight.data)

with torch.no_grad():
    actual_output = custom_norm(dummy_input)

# Check maximum difference
diff = (expected_output - actual_output).abs().max().item()
assert torch.allclose(expected_output, actual_output, atol=1e-5, rtol=1e-4), f"Mismatch! Max diff: {diff}"
print(f"[+] Layer verified! Max diff: {diff:.8f}")
