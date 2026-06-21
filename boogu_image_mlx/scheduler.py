"""FlowMatchEulerDiscreteScheduler with Boogu time-shifting, MLX port.

Faithful to boogu/schedulers/scheduling_flow_match_euler_discrete_time_shifting.py.
The schedule (linspace + logistic/v2 time-shift) is pure scalar/numpy math; only
the Euler `step` touches the latent, so it runs on MLX arrays.

Base config: do_shift=True, dynamic_time_shift=False, time_shift_version="v1",
seq_len=4096, base_shift=0.5, max_shift=1.15.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import mlx.core as mx


def _get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def _time_shift_v1(t_np: np.ndarray, mu: float, sigma: float = 1.0) -> np.ndarray:
    eps = 1e-8
    t1 = np.clip(1.0 - t_np, eps, 1.0 - eps)
    num = math.exp(mu)
    denom = num + np.power(1.0 / t1 - 1.0, sigma)
    return (1.0 - num / denom).astype(np.float32)


def _time_shift_v2(t_np: np.ndarray, m: float) -> np.ndarray:
    return (t_np / (m - m * t_np + t_np)).astype(np.float32)


class FlowMatchEulerDiscreteScheduler:
    def __init__(self, num_train_timesteps: int = 1000, do_shift: bool = True,
                 dynamic_time_shift: bool = False, time_shift_version: str = "v1",
                 seq_len: Optional[int] = 4096, base_shift: float = 0.5, max_shift: float = 1.15,
                 time_shift_v2_half_scaling_factor: float = 60.0, **_ignored):
        self.num_train_timesteps = num_train_timesteps
        self.do_shift = do_shift
        self.dynamic_time_shift = dynamic_time_shift
        self.time_shift_version = time_shift_version
        self.seq_len = seq_len
        self.base_shift = base_shift
        self.max_shift = max_shift
        self.time_shift_v2_scaling_factor = time_shift_v2_half_scaling_factor * 2
        self.timesteps = None
        self._timesteps = None
        self._step_index = None

    @classmethod
    def from_config(cls, cfg: dict) -> "FlowMatchEulerDiscreteScheduler":
        return cls(**{k: v for k, v in cfg.items() if not k.startswith("_")})

    def set_timesteps(self, num_inference_steps: int, num_tokens: Optional[int] = None):
        t_arr = np.linspace(0, 1, num_inference_steps + 1, dtype=np.float32)[:-1]
        if self.do_shift:
            if self.dynamic_time_shift:
                if self.time_shift_version == "v1" and num_tokens:
                    tokens_reduced = max(1, int(num_tokens) // 4)
                    mu = _get_lin_function(y1=self.base_shift, y2=self.max_shift)(tokens_reduced)
                    t_arr = _time_shift_v1(t_arr, mu, sigma=1.0)
                elif self.time_shift_version == "v2" and num_tokens:
                    m = float(np.sqrt(num_tokens)) / self.time_shift_v2_scaling_factor
                    t_arr = _time_shift_v2(t_arr, m)
            else:
                if self.time_shift_version == "v1" and self.seq_len:
                    mu = _get_lin_function(y1=self.base_shift, y2=self.max_shift)(int(self.seq_len))
                    t_arr = _time_shift_v1(t_arr, mu, sigma=1.0)
                elif self.time_shift_version == "v2" and self.seq_len:
                    m = float(np.sqrt(self.seq_len)) / self.time_shift_v2_scaling_factor
                    t_arr = _time_shift_v2(t_arr, m)
        self.timesteps = t_arr.astype(np.float32)
        self._timesteps = np.concatenate([self.timesteps, np.ones(1, dtype=np.float32)])
        self._step_index = 0
        return self.timesteps

    def step(self, model_output: mx.array, step_index: int, sample: mx.array) -> mx.array:
        """Euler flow step. `step_index` indexes self.timesteps (explicit, no state)."""
        t = float(self._timesteps[step_index])
        t_next = float(self._timesteps[step_index + 1])
        return sample + (t_next - t) * model_output
