"""Parity for the flow-match + time-shift scheduler vs the Boogu oracle."""

import json
import os
import sys

import numpy as np
import mlx.core as mx

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from boogu_image_mlx.scheduler import FlowMatchEulerDiscreteScheduler  # noqa: E402

ORACLE = os.path.expanduser("~/Development/mlxengine-image/boogu-oracle")


def main() -> int:
    base = os.path.expanduser(os.environ.get("BOOGU_BASE",
            "~/Development/mlxengine-image/weights/Boogu-Image-0.1-Base"))
    cfg = json.load(open(os.path.join(base, "scheduler", "scheduler_config.json")
                        if os.path.exists(os.path.join(base, "scheduler", "scheduler_config.json"))
                        else os.path.join(base, "scheduler",
                             [f for f in os.listdir(os.path.join(base, "scheduler")) if f.endswith(".json")][0])))
    mlx_sch = FlowMatchEulerDiscreteScheduler.from_config(cfg)
    mlx_ts = mlx_sch.set_timesteps(50)

    # Oracle
    sys.path.insert(0, ORACLE)
    import torch
    from boogu.schedulers.scheduling_flow_match_euler_discrete_time_shifting import (
        FlowMatchEulerDiscreteScheduler as TorchSched,
    )
    tsch = TorchSched(**{k: v for k, v in cfg.items() if not k.startswith("_")})
    tsch.set_timesteps(50)
    pt_ts = tsch.timesteps.cpu().numpy()

    d = np.abs(pt_ts - mlx_ts)
    ok = d.max() < 1e-6
    print(f"[timesteps] {'PASS' if ok else 'FAIL'} max_abs={d.max():.3e} "
          f"first3 pt={pt_ts[:3]} mx={mlx_ts[:3]}")

    # step parity at a few indices
    rng = np.random.default_rng(0)
    sample = rng.standard_normal((1, 16, 16, 16)).astype(np.float32)
    mout = rng.standard_normal((1, 16, 16, 16)).astype(np.float32)
    step_ok = True
    for idx in [0, 25, 49]:
        mx_prev = np.array(mlx_sch.step(mx.array(mout), idx, mx.array(sample)))
        tsch._step_index = idx
        pt_prev = tsch.step(torch.from_numpy(mout), torch.tensor(float(pt_ts[idx])),
                            torch.from_numpy(sample), return_dict=False)[0].cpu().numpy()
        dd = np.abs(pt_prev - mx_prev).max()
        if dd >= 1e-6:
            step_ok = False
        print(f"[step {idx}] {'PASS' if dd < 1e-6 else 'FAIL'} max_abs={dd:.3e}")

    return 0 if (ok and step_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
