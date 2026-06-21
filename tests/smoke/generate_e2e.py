"""First end-to-end Boogu-Image generation (MLX).

Conditioning golden (torch mllm) -> MLX DiT denoise (CFG) -> MLX VAE decode -> PNG.
Runs the DiT in bf16 on the default device (GPU); VAE decode in fp32.

    BOOGU_BASE=.../Boogu-Image-0.1-Base \
    .venv/bin/python tests/smoke/generate_e2e.py
"""

import json
import os
import sys
import time

import numpy as np
import mlx.core as mx

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from boogu_image_mlx.models.transformer import BooguImageTransformer2DModel  # noqa: E402
from boogu_image_mlx.models.vae import AutoencoderKL  # noqa: E402
from boogu_image_mlx.scheduler import FlowMatchEulerDiscreteScheduler  # noqa: E402
from boogu_image_mlx.utils.weights import (  # noqa: E402
    read_safetensors_dir, read_safetensors_np, load_named_into_mlx, load_diffusers_into_mlx)

BASE = os.path.expanduser(os.environ.get("BOOGU_BASE",
        "~/Development/mlxengine-image/weights/Boogu-Image-0.1-Base"))
H = int(os.environ.get("BOOGU_H", "512"))
W = int(os.environ.get("BOOGU_W", "512"))
STEPS = int(os.environ.get("BOOGU_STEPS", "24"))
GUIDANCE = float(os.environ.get("BOOGU_GUIDANCE", "3.0"))
SEED = int(os.environ.get("BOOGU_SEED", "0"))
GOLDEN = os.path.join(ROOT, "tests", "parity", "conditioning_golden.npz")
OUT = os.path.join(ROOT, "samples", "first_e2e.png")


def main() -> int:
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tcfg = json.load(open(os.path.join(BASE, "transformer", "config.json")))
    vcfg = json.load(open(os.path.join(BASE, "vae", "config.json")))
    scfg = json.load(open(os.path.join(BASE, "scheduler", "scheduler_config.json")))

    # --- models ---
    print("loading DiT (bf16)...")
    dit = BooguImageTransformer2DModel.from_config(tcfg)
    load_named_into_mlx(dit, read_safetensors_dir(os.path.join(BASE, "transformer"), dtype=mx.bfloat16))
    print("loading VAE (fp32)...")
    vae = AutoencoderKL.from_config(vcfg)
    load_diffusers_into_mlx(vae, read_safetensors_np(
        os.path.join(BASE, "vae", "diffusion_pytorch_model.safetensors")))
    sched = FlowMatchEulerDiscreteScheduler.from_config(scfg)

    # --- conditioning ---
    g = np.load(GOLDEN, allow_pickle=True)
    pos = mx.array(g["feats"]).astype(mx.bfloat16)        # [1, Lp, 4096]
    neg = mx.array(g["feats_neg"]).astype(mx.bfloat16)    # [1, Ln, 4096]
    print(f"prompt: {str(g['prompt'])!r} | pos {pos.shape} neg {neg.shape}")

    # --- latents ---
    hl, wl = H // 8, W // 8
    sched.set_timesteps(STEPS, num_tokens=hl * wl)
    mx.random.seed(SEED)
    lat = mx.random.normal((1, 16, hl, wl)).astype(mx.bfloat16)

    t0 = time.time()
    for i in range(STEPS):
        t = mx.array([float(sched.timesteps[i])], dtype=mx.bfloat16)
        pred_c = dit(lat, t, pos)
        if GUIDANCE > 1.0:
            pred_u = dit(lat, t, neg)
            pred = pred_u + GUIDANCE * (pred_c - pred_u)
        else:
            pred = pred_c
        lat = sched.step(pred, i, lat)
        mx.eval(lat)
        mx.clear_cache()
        print(f"  step {i+1}/{STEPS}", end="\r")
    print(f"\ndenoise: {time.time()-t0:.1f}s")

    # --- decode ---
    z = lat.astype(mx.float32) / vcfg["scaling_factor"] + vcfg["shift_factor"]
    img = vae.decode(z)                                    # [1,3,H,W], ~[-1,1]
    mx.eval(img)
    arr = np.array(img)[0]                                 # [3,H,W]
    arr = np.clip(arr / 2 + 0.5, 0, 1)
    arr = (arr.transpose(1, 2, 0) * 255).astype(np.uint8)  # HWC

    from PIL import Image
    Image.fromarray(arr).save(OUT)
    finite = np.isfinite(arr).all()
    print(f"saved {OUT} | shape {arr.shape} | finite {finite} | "
          f"mean {arr.mean():.1f} std {arr.std():.1f}")
    return 0 if finite and arr.std() > 2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
