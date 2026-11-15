"""
Phase 1: Foundations of Inference Engineering
Covers:
- System hardware audit (CPU, RAM, GPU, Disk)
- Tensor shapes, indexing, and dimensions [Batch, SeqLen, HiddenDim]
- Data types & precision memory math (FP32 vs FP16 vs BF16)
- Matrix multiplication (Linear projections in LLMs)
- Security audit: Safe model loading with Safetensors (Preventing CWE-502)
"""

import os
import platform
import shutil
import psutil
import torch


def audit_system_environment() -> dict:
    """Audit hardware resources available for LLM inference."""
    ram = psutil.virtual_memory()
    total_disk, _, free_disk = shutil.disk_usage(".")
    
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU Only"
    vram_gb = (
        torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if cuda_available
        else 0.0
    )

    info = {
        "os": f"{platform.system()} {platform.release()}",
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_total_gb": round(ram.total / (1024**3), 2),
        "ram_available_gb": round(ram.available / (1024**3), 2),
        "disk_free_gb": round(free_disk / (1024**3), 2),
        "cuda_available": cuda_available,
        "device_name": device_name,
        "vram_gb": round(vram_gb, 2),
    }
    return info


def demonstrate_tensors_and_shapes() -> None:
    """
    Demonstrate LLM tensor shapes and indexing.
    Inference standard layout: [Batch_Size, Sequence_Length, Hidden_Dimension]
    """
    batch_size = 2       # 2 independent user requests
    seq_len = 4          # 4 tokens per request
    hidden_dim = 8       # 8-dimensional representation vector per token

    # Construct synthetic activation tensor
    activations = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float32)

    print(f"Activations Tensor Shape: {activations.shape}")
    print(f"- Batch Size (concurrent prompts): {activations.shape[0]}")
    print(f"- Sequence Length (tokens per prompt): {activations.shape[1]}")
    print(f"- Hidden Dimension (features per token): {activations.shape[2]}")

    # Inspect the representation of the first token of the first prompt
    first_token_vector = activations[0, 0, :]
    print(f"First token representation shape: {first_token_vector.shape}")


def demonstrate_precision_and_memory() -> None:
    """
    Compare FP32, FP16, and BF16.
    Calculates exact memory footprint for weight storage and activations.
    """
    num_elements = 1_000_000  # 1 Million parameters

    t_fp32 = torch.zeros(num_elements, dtype=torch.float32)
    t_fp16 = torch.zeros(num_elements, dtype=torch.float16)
    t_bf16 = torch.zeros(num_elements, dtype=torch.bfloat16)

    # Element size in bytes
    fp32_mb = (t_fp32.nelement() * t_fp32.element_size()) / (1024**2)
    fp16_mb = (t_fp16.nelement() * t_fp16.element_size()) / (1024**2)
    bf16_mb = (t_bf16.nelement() * t_bf16.element_size()) / (1024**2)

    print(f"1M parameters in FP32 : {fp32_mb:.2f} MB ({t_fp32.element_size()} bytes/param)")
    print(f"1M parameters in FP16 : {fp16_mb:.2f} MB ({t_fp16.element_size()} bytes/param)")
    print(f"1M parameters in BF16 : {bf16_mb:.2f} MB ({t_bf16.element_size()} bytes/param)")


def demonstrate_linear_projection() -> None:
    """
    Understand Matrix Multiplication (Linear Projection) in LLMs.
    Every Transformer layer projects input vectors into Query, Key, and Value vectors.
    Operation: Y = X @ W^T
    """
    batch_size = 1
    seq_len = 3
    in_features = 4    # Current hidden dimension
    out_features = 6   # Projected dimension

    x = torch.randn(batch_size, seq_len, in_features)
    weight = torch.randn(out_features, in_features)  # [out_features, in_features]

    # Matrix multiplication: [1, 3, 4] @ [4, 6] -> [1, 3, 6]
    output = x @ weight.T

    # Numerical verification against PyTorch nn.functional.linear
    expected = torch.nn.functional.linear(x, weight, bias=None)
    assert torch.allclose(output, expected, atol=1e-6), "MatMul mismatch with nn.functional.linear"
    print(f"Projection Input Shape: {x.shape} -> Output Shape: {output.shape} (Verified)")


def security_note_safe_loading() -> None:
    """
    Security check: Explain arbitrary code execution risks in weight serialization.
    """
    print("""
    ============================== SECURITY CHECK ==============================
    Vulnerability: CWE-502 (Deserialization of Untrusted Data)
    - Legacy PyTorch (.pt/.bin/.pkl) uses Python's pickle module.
    - An attacker can craft a malicious pickle payload that executes arbitrary
      system commands during torch.load(weights_path).
    Fix:
    - Use Safetensors (.safetensors). It is pure tensor metadata + raw bytes.
    - Zero arbitrary code execution risk, zero-copy loading via memory mapping (mmap).
    - Always set `trust_remote_code=False` unless the repository is strictly vetted.
    ============================================================================
    """)


if __name__ == "__main__":
    print("=== [1] Hardware & Environment Audit ===")
    env = audit_system_environment()
    for k, v in env.items():
        print(f"  {k}: {v}")

    print("\n=== [2] Tensor Shapes & Indexing ===")
    demonstrate_tensors_and_shapes()

    print("\n=== [3] Precision & Memory Math ===")
    demonstrate_precision_and_memory()

    print("\n=== [4] Matrix Multiplication (Linear Projection) ===")
    demonstrate_linear_projection()

    security_note_safe_loading()
