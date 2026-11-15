from transformers import AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen3-0.6B"

print("Loading Qwen3...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype="float32"
)

print("\nFIRST TRANSFORMER LAYER")
print(model.model.layers[0])

print("\nFIRST LAYER NORMALIZATION")
print(model.model.layers[0].input_layernorm)

print("\nFIRST LAYER ATTENTION")
print(model.model.layers[0].self_attn)

print("\nFIRST LAYER MLP")
print(model.model.layers[0].mlp)