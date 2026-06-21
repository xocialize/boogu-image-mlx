"""PT-vs-MLX parity for the FLUX AutoencoderKL port.

Run on the CPU stream (Apple-GPU fp32 matmul noise masks real op bugs). Gates:
encode moments and decode output both < 1e-3 max_abs at fp32.

Usage:
    BOOGU_BASE=~/Development/mlxengine-image/weights/Boogu-Image-0.1-Base \
    .venv/bin/python tests/parity/test_vae_parity.py
"""

import json
import os
import sys

import mlx.core as mx
import numpy as np

mx.set_default_device(mx.cpu)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from boogu_image_mlx.models.vae import AutoencoderKL  # noqa: E402
from boogu_image_mlx.utils.weights import load_diffusers_into_mlx, read_safetensors_np  # noqa: E402
from _helpers import make_seeded_input  # noqa: E402


def main() -> int:
    base = os.path.expanduser(os.environ.get("BOOGU_BASE",
            "~/Development/mlxengine-image/weights/Boogu-Image-0.1-Base"))
    vae_dir = os.path.join(base, "vae")
    cfg = json.load(open(os.path.join(vae_dir, "config.json")))
    st_path = os.path.join(vae_dir, "diffusion_pytorch_model.safetensors")

    # --- MLX model ---
    mlx_vae = AutoencoderKL.from_config(cfg)
    sd = read_safetensors_np(st_path)
    loaded = load_diffusers_into_mlx(mlx_vae, sd)

    # coverage check
    mlx_keys = {k for k, _ in __import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten(mlx_vae.parameters())}
    missing = mlx_keys - set(loaded)
    extra = set(loaded) - mlx_keys
    print(f"weights: {len(loaded)} loaded | model params {len(mlx_keys)} | missing {len(missing)} | extra {len(extra)}")
    if missing:
        print("  MISSING:", sorted(missing)[:10])
    if extra:
        print("  EXTRA:", sorted(extra)[:10])

    # --- inputs (small spatial size for a fast CPU run) ---
    img = make_seeded_input((1, 3, 64, 64), seed=42)          # encode input
    lat = make_seeded_input((1, cfg["latent_channels"], 16, 16), seed=7)  # decode input

    # --- MLX forward ---
    mx_moments = np.array(mlx_vae.encode_moments(mx.array(img)))
    mx_decoded = np.array(mlx_vae.decode(mx.array(lat)))

    # --- PyTorch oracle ---
    import torch
    from diffusers import AutoencoderKL as TorchVAE

    tv = TorchVAE.from_pretrained(vae_dir).to(torch.float32).eval()
    with torch.no_grad():
        pt_moments = tv.encode(torch.from_numpy(img)).latent_dist.parameters.cpu().numpy()
        pt_decoded = tv.decode(torch.from_numpy(lat)).sample.cpu().numpy()

    ok = True
    for name, a, b, thr in [
        ("encode_moments", pt_moments, mx_moments, 1e-3),
        ("decode", pt_decoded, mx_decoded, 1e-3),
    ]:
        if a.shape != b.shape:
            print(f"[{name}] SHAPE MISMATCH pt={a.shape} mx={b.shape}")
            ok = False
            continue
        d = np.abs(a - b)
        status = "PASS" if d.max() < thr else "FAIL"
        if d.max() >= thr:
            ok = False
        print(f"[{name}] {status} max_abs={d.max():.3e} mean_abs={d.mean():.3e} "
              f"pt_range=({a.min():.3f},{a.max():.3f})")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
