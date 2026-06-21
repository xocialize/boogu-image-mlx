# boogu-image-mlx

Pure [MLX](https://github.com/ml-explore/mlx) port of **[Boogu-Image-0.1](https://huggingface.co/Boogu/Boogu-Image-0.1-Base)** (Apache-2.0) for Apple Silicon — text-to-image and instruction editing.

## Architecture

Boogu-Image is an **OmniGen2-lineage** (BAAI; NextDiT/Lumina2-derived) pipeline:

| Component | What | Port status |
|---|---|---|
| `mllm` | Qwen3-VL-8B-Instruct conditioner (hidden-feature extraction, mean-reduce, dim 4096) | reuse `mlx-community/Qwen3-VL-8B` |
| `vae` | FLUX.1 `AutoencoderKL`, 16-ch (scaling 0.3611, shift 0.1159) | ✅ ported, parity-locked |
| `scheduler` | `FlowMatchEulerDiscreteScheduler` + time-shifting | in progress |
| `transformer` | `BooguImageTransformer2DModel` — 8 double-stream + 32 single-stream + refiners, GQA 28h/7kv, 3-axis RoPE | in progress |

## Parity testing

PT-vs-MLX parity runs on the CPU stream (`mx.set_default_device(mx.cpu)`); gate < 1e-3 max_abs at fp32.

```bash
uv pip install -e ".[parity]"
BOOGU_BASE=/path/to/Boogu-Image-0.1-Base .venv/bin/python tests/parity/test_vae_parity.py
```

VAE parity (latest): `encode_moments` max_abs 1.97e-4, `decode` max_abs 6.7e-6.

## Status

Base (T2I) port in progress. Turbo (4-step DMD) and Edit variants follow.
