import os
import torch
from transformers import AutoModelForCausalLM
from torch.profiler import profile, record_function, ProfilerActivity

# -------------------------------------------------------------------------
# 1. Architecture Component: CustomRMSNorm
# -------------------------------------------------------------------------
class CustomRMSNorm(torch.nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    Normalizes activations across the hidden dimension without subtracting the mean.
    Formula: y = (x / sqrt(mean(x^2) + eps)) * weight
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        normed = x * torch.rsqrt(variance + self.eps)
        return normed * self.weight


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Running profiling on target device: {device.upper()}")

    # ---------------------------------------------------------------------
    # 2. Reference Model Layer (Qwen2.5-1.5B)
    # ---------------------------------------------------------------------
    model_id = "Qwen/Qwen2.5-1.5B"
    print(f"[*] Loading reference layer from {model_id}...")
    
    # Security: trust_remote_code=False prevents arbitrary code execution
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=False
    )
    hf_model.eval()

    hf_norm = hf_model.model.layers[0].input_layernorm.to(device)
    hidden_dim = hf_norm.weight.shape[0]

    # Instantiate custom implementation & sync weights
    custom_norm = CustomRMSNorm(hidden_dim, eps=hf_norm.variance_epsilon).to(device)
    custom_norm.weight.data.copy_(hf_norm.weight.data)
    custom_norm.eval()

    # Synthetic batch: [batch_size=2, seq_len=512, hidden_dim]
    dummy_input = torch.randn(2, 512, hidden_dim, device=device)

    # Verification: check numerical parity before profiling
    with torch.no_grad():
        expected_out = hf_norm(dummy_input)
        actual_out = custom_norm(dummy_input)
        diff = (expected_out - actual_out).abs().max().item()
        assert torch.allclose(expected_out, actual_out, atol=1e-5, rtol=1e-4), f"Mismatch detected: {diff}"
        print(f"[+] Numerical verification passed! Max diff: {diff:.8f}")

    # ---------------------------------------------------------------------
    # 3. Perfetto Profiling with PyTorch Profiler
    # ---------------------------------------------------------------------
    trace_output_file = "perfetto_trace.json"
    
    # Configure profiler activities (CPU and GPU if available)
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    print("[*] Starting PyTorch Profiler...")

    # Warmup runs to eliminate JIT compilation, cache loading, and lazy init noise
    with torch.no_grad():
        for _ in range(5):
            _ = custom_norm(dummy_input)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Profile with explicit instrumentation labels
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with torch.no_grad():
            for step in range(10):
                # Tagging operations with record_function creates distinct slices in Perfetto
                with record_function(f"step_{step}_custom_rmsnorm"):
                    with record_function("compute_variance"):
                        variance = dummy_input.pow(2).mean(dim=-1, keepdim=True)
                    with record_function("compute_rsqrt"):
                        normed = dummy_input * torch.rsqrt(variance + custom_norm.eps)
                    with record_function("scale_weight"):
                        _ = normed * custom_norm.weight

                with record_function(f"step_{step}_huggingface_rmsnorm"):
                    _ = hf_norm(dummy_input)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Export trace for Perfetto (ui.perfetto.dev)
    prof.export_chrome_trace(trace_output_file)
    print(f"[+] Profiling complete! Trace saved to: {os.path.abspath(trace_output_file)}")
    print("[*] To view trace: Open https://ui.perfetto.dev and drag & drop the generated JSON file.")

    # Print summary table in console
    print("\n" + "=" * 60)
    print("Top 10 Time-Consuming Operations (Self CPU Time):")
    print("=" * 60)
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))


if __name__ == "__main__":
    main()
