"""
Phase 2: Loading and Inspecting the Reference Model (Qwen3)
Covers:
- Safe model and tokenizer loading (trust_remote_code=False, safetensors)
- Inspecting configuration (hidden dimension, heads, layers, vocabulary)
- Tokenizer mechanics (encoding text to token IDs and decoding back)
- Basic text generation and inference timing
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(model_id: str = "Qwen/Qwen3-0.6B"):
    """
    Safely load a pretrained model and tokenizer.
    
    Security:
    - trust_remote_code=False: Prevents arbitrary code execution from remote repositories.
    - use_safetensors=True: Enforces loading from safe binary tensors rather than Python pickles.
    """
    # 1. Determine device; keep weights in float32
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    dtype = torch.float32

    print(f"[*] Target Device: {device.upper()} | Data Type: {dtype}")
    print(f"[*] Loading Tokenizer for {model_id}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=False
    )

    print(f"[*] Loading Model Weights for {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map=device,
        trust_remote_code=False,
        use_safetensors=True
    )
    
    # Put model strictly in evaluation mode (disables dropout, freezes weights)
    model.eval()
    
    return model, tokenizer, device


def inspect_model_architecture(model, tokenizer) -> None:
    """Inspect model configuration, parameters, and layer hierarchy."""
    cfg = model.config
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "=" * 50)
    print("      MODEL ARCHITECTURE & CONFIGURATION")
    print("=" * 50)
    print(f"Model Architecture       : {cfg.architectures[0]}")
    print(f"Vocabulary Size          : {cfg.vocab_size:,} tokens")
    print(f"Hidden Dimension (d_model): {cfg.hidden_size}")
    print(f"Number of Layers         : {cfg.num_hidden_layers}")
    print(f"Attention Query Heads    : {cfg.num_attention_heads}")
    print(f"Attention Key/Value Heads: {cfg.num_key_value_heads} (GQA factor: {cfg.num_attention_heads // cfg.num_key_value_heads})")
    print(f"Intermediate Dimension   : {cfg.intermediate_size} (FFN / SwiGLU)")
    print(f"Max Position Embeddings  : {cfg.max_position_embeddings:,} tokens")
    print(f"Total Parameters         : {num_params:,} ({num_params / 1e9:.2f}B)")
    print(f"Trainable Parameters     : {num_trainable}")
    print("=" * 50)


def demonstrate_tokenizer(tokenizer) -> torch.Tensor:
    """Demonstrate how text turns into token IDs."""
    prompt = "Inference engineering is"
    print(f"\nOriginal text: '{prompt}'")

    # Encode: Text -> List of IDs
    tokens = tokenizer.encode(prompt, return_tensors="pt")
    print(f"Token IDs shape: {tokens.shape}")
    print(f"Token IDs tensor: {tokens}")

    # Inspect token-by-token breakdown
    print("\nToken-by-Token Breakdown:")
    for token_id in tokens[0]:
        token_str = tokenizer.decode([token_id])
        print(f"  ID {token_id.item():>6} -> '{token_str}'")

    # Decode: IDs -> Reconstructed Text
    decoded_text = tokenizer.decode(tokens[0], skip_special_tokens=True)
    print(f"\nDecoded text: '{decoded_text}'")
    return tokens


def run_simple_generation(model, tokenizer, device, prompt: str = "Explain inference engineering in one sentence."):
    """Run baseline generation and record latency and tokens per second."""
    # Instruct models use a structured chat template
    messages = [
        {"role": "system", "content": "You are a helpful AI engineer."},
        {"role": "user", "content": prompt}
    ]
    
    # Format according to Qwen chat template
    text_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text_prompt, return_tensors="pt").to(device)

    input_tokens_count = inputs["input_ids"].shape[1]

    print(f"\n[*] Generating response for prompt: \"{prompt}\"")
    start_time = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,        # Greedy decoding for deterministic benchmark
            temperature=None,
            top_p=None
        )

    end_time = time.perf_counter()
    latency = end_time - start_time

    # Calculate tokens per second (generated tokens only)
    total_tokens = outputs.shape[1]
    new_tokens_count = total_tokens - input_tokens_count
    tokens_per_sec = new_tokens_count / latency if latency > 0 else 0

    response = tokenizer.decode(outputs[0][input_tokens_count:], skip_special_tokens=True)

    print(f"\n--- Output Response ---")
    print(response.strip())
    print("-" * 50)
    print(f"Input Tokens     : {input_tokens_count}")
    print(f"Generated Tokens : {new_tokens_count}")
    print(f"Latency          : {latency:.2f} seconds")
    print(f"Generation Speed : {tokens_per_sec:.2f} tokens/second")
    print("-" * 50)


if __name__ == "__main__":
    MODEL_ID = "Qwen/Qwen3-0.6B"
    
    model, tokenizer, device = load_model_and_tokenizer(MODEL_ID)
    inspect_model_architecture(model, tokenizer)
    demonstrate_tokenizer(tokenizer)
    run_simple_generation(model, tokenizer, device)
