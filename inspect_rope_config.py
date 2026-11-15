from transformers import AutoConfig

config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")

print("Model type:", config.model_type)
print("Hidden size:", config.hidden_size)
print("Attention heads:", config.num_attention_heads)
print("KV heads:", config.num_key_value_heads)
print("Head dimension:", config.head_dim)
print("RoPE theta:", config.rope_theta)
print("RoPE scaling:", config.rope_scaling)
print("Max positions:", config.max_position_embeddings)