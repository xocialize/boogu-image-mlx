"""Quantize the DiT to int4 and gate on per-pass cosine vs bf16 (CPU stream).

Quantizes only attention + feed-forward Linears (group_size=32 — 3360 is not
divisible by 64); keeps x_embedder / time_caption_embed / norm_out / AdaLN
(*norm*.linear) at bf16. Saves the quantized weights for packaging.
"""

import json
import os
import sys

import numpy as np
import mlx.core as mx
import mlx.nn as nn

# Quant graph grinds on the CPU stream; run forwards on GPU (cosine gate is
# precision-robust). Lesson: quantize on CPU/load, forward on GPU.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from boogu_image_mlx.models.transformer import BooguImageTransformer2DModel  # noqa: E402
from boogu_image_mlx.utils.weights import read_safetensors_dir, load_named_into_mlx  # noqa: E402
from _helpers import make_seeded_input  # noqa: E402

GROUP, BITS = 32, 4


def quant_predicate(path: str, module) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    if not (("attn" in path) or ("feed_forward" in path)):  # keep embeds/time/norm_out/AdaLN
        return False
    return module.weight.shape[1] % GROUP == 0


def main() -> int:
    base = os.path.expanduser(os.environ.get("BOOGU_BASE",
            "~/Development/mlxengine-image/weights/Boogu-Image-0.1-Base"))
    tdir = os.path.join(base, "transformer")
    cfg = json.load(open(os.path.join(tdir, "config.json")))

    dit = BooguImageTransformer2DModel.from_config(cfg)
    load_named_into_mlx(dit, read_safetensors_dir(tdir, dtype=mx.bfloat16))

    g = np.load(os.path.join(HERE, "conditioning_golden.npz"), allow_pickle=True)
    latent = mx.array(make_seeded_input((1, 16, 16, 16), seed=1)).astype(mx.bfloat16)
    ts = mx.array(np.array([0.5], dtype=np.float32)).astype(mx.bfloat16)
    instr = mx.array(g["feats"][:, :16]).astype(mx.bfloat16)

    out_bf16 = np.array(dit(latent, ts, instr).astype(mx.float32)).reshape(-1)

    nn.quantize(dit, group_size=GROUP, bits=BITS, class_predicate=quant_predicate)
    n_q = sum(1 for _, m in dit.named_modules() if isinstance(m, nn.QuantizedLinear))
    out_int4 = np.array(dit(latent, ts, instr).astype(mx.float32)).reshape(-1)

    cos = float(out_bf16 @ out_int4 / (np.linalg.norm(out_bf16) * np.linalg.norm(out_int4) + 1e-8))
    print(f"quantized {n_q} Linears (group={GROUP} bits={BITS}) | per-pass cos={cos:.5f} "
          f"-> {'PASS' if cos >= 0.99 else 'FAIL'}")

    if cos >= 0.99 and os.environ.get("BOOGU_SAVE_INT4"):
        from mlx.utils import tree_flatten
        outdir = os.environ["BOOGU_SAVE_INT4"]
        os.makedirs(outdir, exist_ok=True)
        flat = dict(tree_flatten(dit.parameters()))
        mx.save_safetensors(os.path.join(outdir, "transformer_int4.safetensors"), flat)
        print("saved int4 weights ->", outdir)
    return 0 if cos >= 0.99 else 1


if __name__ == "__main__":
    raise SystemExit(main())
