"""Pure-MLX Boogu-Image T2I pipeline.

Qwen3-VL (mlx-vlm) instruction features -> BooguImageTransformer2DModel denoise
(CFG) -> FLUX AutoencoderKL decode. No torch at inference. The Qwen3-VL encoder
is the stock model referenced from mlx-community/Qwen3-VL-8B-Instruct.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import mlx.core as mx

from .models.transformer import BooguImageTransformer2DModel
from .models.vae import AutoencoderKL
from .scheduler import FlowMatchEulerDiscreteScheduler
from .utils.weights import (read_safetensors_dir, read_safetensors_np,
                            load_named_into_mlx, load_diffusers_into_mlx)

SYSTEM_T2I = ("You are a helpful assistant that generates high-quality images based "
              "on user instructions. The instructions are as follows.")
SYSTEM_TI2I = ("Describe the key features of the input image (color, shape, size, texture, objects, "
               "background), then explain how the user's text instruction should alter or modify the "
               "image. Generate a new image that meets the user's requirements while maintaining "
               "consistency with the original input where appropriate.")


class _Identity:
    def __call__(self, x):
        return x


class BooguImagePipeline:
    def __init__(self, dit, vae, scheduler, qwen_model, qwen_processor, vae_cfg):
        self.dit = dit
        self.vae = vae
        self.scheduler = scheduler
        self.qwen = qwen_model
        self.processor = qwen_processor
        self.vae_cfg = vae_cfg

    @classmethod
    def from_pretrained(cls, base_dir: str, qwen_dir: str, dit_dtype=mx.bfloat16):
        import mlx.nn as nn
        base_dir = os.path.expanduser(base_dir)
        tdir = os.path.join(base_dir, "transformer")
        tcfg = json.load(open(os.path.join(tdir, "config.json")))
        vcfg = json.load(open(os.path.join(base_dir, "vae", "config.json")))
        scfg = json.load(open(os.path.join(base_dir, "scheduler", "scheduler_config.json")))

        dit = BooguImageTransformer2DModel.from_config(tcfg)
        qpath = os.path.join(tdir, "quant_config.json")
        qc = json.load(open(qpath)) if os.path.exists(qpath) else None
        wfile = os.path.join(tdir, qc.get("weights_file", "transformer_int4.safetensors")) if qc else ""
        if qc and os.path.exists(wfile):
            g, b = qc["group_size"], qc["bits"]

            def _pred(path, m):
                return (isinstance(m, nn.Linear)
                        and (("attn" in path) or ("feed_forward" in path))
                        and m.weight.shape[1] % g == 0)
            nn.quantize(dit, group_size=g, bits=b, class_predicate=_pred)
            load_named_into_mlx(dit, {k: v for k, v in mx.load(wfile).items()})
        else:
            load_named_into_mlx(dit, read_safetensors_dir(tdir, dtype=dit_dtype))
        vae = AutoencoderKL.from_config(vcfg)
        load_diffusers_into_mlx(vae, read_safetensors_np(
            os.path.join(base_dir, "vae", "diffusion_pytorch_model.safetensors")))
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(scfg)

        from mlx_vlm import load as vlm_load
        qwen_model, processor = vlm_load(os.path.expanduser(qwen_dir))
        pipe = cls(dit, vae, scheduler, qwen_model, processor, vcfg)
        # HF processor for image preprocessing (PIL -> pixel_values); proven path.
        pipe._proc_dir = os.path.join(base_dir, "processor")
        pipe._hf_processor = None
        return pipe

    @property
    def hf_processor(self):
        if self._hf_processor is None:
            from transformers import AutoProcessor
            self._hf_processor = AutoProcessor.from_pretrained(self._proc_dir)
        return self._hf_processor

    def encode_prompt(self, text: str) -> mx.array:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_T2I}]},
            {"role": "user", "content": [{"type": "text", "text": text}]},
        ]
        enc = self.processor.apply_chat_template(
            [messages], tokenize=True, return_dict=True,
            add_generation_prompt=False, padding=True, return_tensors="np")
        ids = mx.array(np.asarray(enc["input_ids"]))
        lm = self.qwen.language_model
        base = lm.model if hasattr(lm, "model") else lm
        feats = base(ids)                                  # [1, L, 4096] last_hidden_state
        return feats.astype(self.dit.x_embedder.weight.dtype)

    def encode_prompt_with_image(self, image, text: str) -> mx.array:
        """TI2I conditioning: Qwen3-VL last_hidden_state over [image, text]."""
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_TI2I}]},
            {"role": "user", "content": [{"type": "image", "image": image},
                                         {"type": "text", "text": text}]},
        ]
        enc = self.hf_processor.apply_chat_template(
            [messages], tokenize=True, return_dict=True,
            add_generation_prompt=False, padding=True, return_tensors="np")
        orig = self.qwen.language_model.lm_head
        self.qwen.language_model.lm_head = _Identity()
        h = self.qwen(mx.array(np.asarray(enc["input_ids"])),
                      pixel_values=mx.array(np.asarray(enc["pixel_values"])),
                      image_grid_thw=mx.array(np.asarray(enc["image_grid_thw"])), mask=None)
        self.qwen.language_model.lm_head = orig
        h = h.logits if hasattr(h, "logits") else h
        return h.astype(self.dit.x_embedder.weight.dtype)

    def vae_encode(self, image, height: int, width: int) -> mx.array:
        """Encode an input image to a scaled ref latent (= (mean - shift) * scale)."""
        img = image.convert("RGB").resize((width, height))
        arr = np.asarray(img).astype(np.float32) / 255.0 * 2 - 1     # [H,W,3] in [-1,1]
        x = mx.array(arr.transpose(2, 0, 1)[None])                   # [1,3,H,W]
        moments = self.vae.encode_moments(x)                         # [1,32,H/8,W/8]
        mean = moments[:, :16]
        ref = (mean - self.vae_cfg["shift_factor"]) * self.vae_cfg["scaling_factor"]
        return ref.astype(self.dit.x_embedder.weight.dtype)

    def generate_edit(self, image, instruction: str, height: int = 768, width: int = 768,
                      steps: int = 50, text_guidance: float = 4.0, seed: int = 0) -> np.ndarray:
        pos = self.encode_prompt_with_image(image, instruction)
        neg = self.encode_prompt_with_image(image, "")
        ref = self.vae_encode(image, height, width)
        hl, wl = height // 8, width // 8
        self.scheduler.set_timesteps(steps, num_tokens=hl * wl)
        mx.random.seed(seed)
        lat = mx.random.normal((1, 16, hl, wl)).astype(pos.dtype)
        for i in range(steps):
            t = mx.array([float(self.scheduler.timesteps[i])], dtype=pos.dtype)
            pc = self.dit(lat, t, pos, ref_latent=ref)
            pu = self.dit(lat, t, neg, ref_latent=ref)
            lat = self.scheduler.step(pu + text_guidance * (pc - pu), i, lat)
            mx.eval(lat); mx.clear_cache()
        z = lat.astype(mx.float32) / self.vae_cfg["scaling_factor"] + self.vae_cfg["shift_factor"]
        img = self.vae.decode(z); mx.eval(img)
        arr = np.clip(np.array(img)[0] / 2 + 0.5, 0, 1)
        return (arr.transpose(1, 2, 0) * 255).astype(np.uint8)

    def generate(self, prompt: str, negative: str = "", height: int = 1024,
                 width: int = 1024, steps: int = 30, guidance: float = 3.5,
                 seed: int = 0) -> np.ndarray:
        pos = self.encode_prompt(prompt)
        neg = self.encode_prompt(negative) if guidance > 1.0 else None

        hl, wl = height // 8, width // 8
        self.scheduler.set_timesteps(steps, num_tokens=hl * wl)
        mx.random.seed(seed)
        lat = mx.random.normal((1, 16, hl, wl)).astype(pos.dtype)

        for i in range(steps):
            t = mx.array([float(self.scheduler.timesteps[i])], dtype=pos.dtype)
            pred = self.dit(lat, t, pos)
            if neg is not None:
                pu = self.dit(lat, t, neg)
                pred = pu + guidance * (pred - pu)
            lat = self.scheduler.step(pred, i, lat)
            mx.eval(lat)
            mx.clear_cache()

        z = lat.astype(mx.float32) / self.vae_cfg["scaling_factor"] + self.vae_cfg["shift_factor"]
        img = self.vae.decode(z)
        mx.eval(img)
        arr = np.clip(np.array(img)[0] / 2 + 0.5, 0, 1)
        return (arr.transpose(1, 2, 0) * 255).astype(np.uint8)   # HWC uint8
