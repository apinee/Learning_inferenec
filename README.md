# Inference, one layer at a time

A small set of scripts that rebuild the pieces of a Transformer decoder, then check them against [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B).

The model reads tokens. Each token is a vector. Qwen3 is a stack of identical layers, and each layer does three things:

1. **RMSNorm** scales the vector so its magnitude stays stable.
2. **Attention** lets each token mix information from earlier tokens.
3. **MLP** lets each token transform its own vector. This part is not implemented here yet.

Most scripts use tiny tensors so the shapes stay readable. Two scripts load a real Hugging Face model and compare a handwritten layer with the library.

## Model used here

`phase2_load_model.py` and the inspection scripts load `Qwen/Qwen3-0.6B` in `float32`.

From the config:

| Setting | Value |
| --- | --- |
| Hidden size | 1024 |
| Layers | 28 |
| Intermediate size | 3072 |
| Query heads | 16 |
| Key/Value heads | 8 |
| Head dimension | 128 |
| Max position | 40960 |
| Vocabulary | 151936 |
| RoPE theta | 1,000,000 |

`RMSNorm.py` and `perfetto.py` still load `Qwen/Qwen2.5-1.5B`. That is enough to check RMSNorm, because the formula is the same.

Weights stay in float32 so the handwritten math can be compared directly with PyTorch. Loading uses `trust_remote_code=False` and safetensors.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The first run that loads a model downloads weights from Hugging Face. CPU is enough for these scripts.

## Read the scripts in this order

### 1. Tensors and a safe load

`phase1_foundations.py` checks the machine, then shows the layout used everywhere else: `[batch, sequence length, hidden size]`. It compares FP32, FP16, and BF16 memory, and shows a linear projection as `Y = X @ W.T`. Query, Key, and Value are that same multiply. The closing note is why weights come from safetensors instead of a pickle file.

`phase2_load_model.py` loads Qwen3-0.6B, prints the config, tokenizes a sentence, and generates a short continuation.

`inspect_qwen3.py` prints one real layer: input norm, self-attention, and MLP. Every later script is one of those boxes, opened up.

### 2. RMSNorm

`RMSNorm.py` implements

```text
y = x * rsqrt(mean(x^2) + eps) * weight
```

and checks the result against Hugging Face on the same input. `perfetto.py` runs the same check, then profiles the custom layer and the library layer with PyTorch Profiler. The trace is written to `perfetto_trace.json` (gitignored). Open it at [ui.perfetto.dev](https://ui.perfetto.dev).

### 3. Attention

Each file adds one piece. The formula underneath all of them is:

```text
scores  = Q @ K.T / sqrt(head_dim)
weights = softmax(scores)
output  = weights @ V
```

| File | What it adds |
| --- | --- |
| `test_attention.py` | One query token attends over three key/value tokens. |
| `test_causal_mask.py` | Future positions are set to `-inf` before the softmax, so a token only sees itself and the past. |
| `test_multihead.py` | Hidden size 8 is split into 2 heads. Heads run attention separately, then concatenate and pass through `o_proj`. |
| `test_gqa.py` | 4 query heads share 2 key/value heads. K and V are repeated so the shapes match. Qwen3-0.6B does this with 16 query heads and 8 KV heads. |

### 4. Position, with RoPE

Attention scores do not know token order. RoPE puts position in by rotating Query and Key. The same vector at two positions becomes two different vectors, and the rotation angle grows with the position.

| File | What it adds |
| --- | --- |
| `test_rope_basic.py` | Rotate one 2D vector by 90 degrees. |
| `test_rope_positions.py` | The same vector `[3, 4]` at positions 0, 1, and 2. Position 0 is a zero angle, so the vector is unchanged. |
| `test_rope_frequencies.py` | Qwen head size. 128 dimensions make 64 pairs, each spinning at its own frequency. `rope_theta` is 1,000,000. Early pairs spin slowly and carry long-range order. Later pairs spin faster. |
| `inspect_rope_config.py` | Reads head size, head counts, and the max position from the real Qwen3 config. |

`test_rope_frequencies.py` rotates neighboring pairs: dimension 0 with 1, 2 with 3. Qwen's `apply_rotary_pos_emb` rotates the first half of the head against the second half (`rotate_half` in `transformers/models/qwen3/modeling_qwen3.py`). The frequencies match. The rotated vectors match only after the pairs are laid out the same way.

## Run one script

```powershell
python phase1_foundations.py
python test_gqa.py
python test_rope_frequencies.py
python phase2_load_model.py
```

`phase2_load_model.py`, `inspect_qwen3.py`, `RMSNorm.py`, and `perfetto.py` download and load full weights. The `test_*.py` files only need PyTorch.

## What is still open

Match the handwritten rotation to `Qwen3RotaryEmbedding` and `apply_rotary_pos_emb`. Then apply that rotation to Q and K inside `test_gqa.py`, before the scores. After attention is complete, the remaining block in `layers[0]` is the MLP (SwiGLU: gate, up, and down projections).
