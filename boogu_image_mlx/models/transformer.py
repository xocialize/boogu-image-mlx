"""BooguImageTransformer2DModel (OmniGen2-lineage) ported to MLX.

Base T2I path (batch=1, no reference images). Module / parameter names mirror
boogu/models/transformers/transformer_boogu.py + block_lumina2.py so the
PyTorch state_dict maps 1:1 (all Linear / RMSNorm — no conv transposes).

Stream topology: x_embedder -> context_refiner(instruct) + noise_refiner(img)
-> 8 double-stream (img<->instruct joint attn) -> fuse -> 32 single-stream
-> norm_out -> unpatchify.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import mlx.core as mx
import mlx.nn as nn


def _silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


# --------------------------------------------------------------------------- #
# RoPE (3-axis, Lumina complex form expressed as real arithmetic)
# --------------------------------------------------------------------------- #
def rope_cos_sin(position_ids: np.ndarray, axes_dim=(40, 40, 40), theta: int = 10000):
    """position_ids: [L, 3] int -> (cos, sin) each [L, sum(axes_dim)//2]."""
    cos_parts, sin_parts = [], []
    for a, dim in enumerate(axes_dim):
        inv_freq = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim))  # [dim/2]
        ang = position_ids[:, a:a + 1].astype(np.float64) * inv_freq[None, :]       # [L, dim/2]
        cos_parts.append(np.cos(ang))
        sin_parts.append(np.sin(ang))
    cos = np.concatenate(cos_parts, axis=-1).astype(np.float32)  # [L, 60]
    sin = np.concatenate(sin_parts, axis=-1).astype(np.float32)
    return cos, sin


def apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """x: [B, L, H, D]; cos/sin: [1, L, 1, D//2]. Complex-pair rotation."""
    B, L, H, D = x.shape
    cos = cos.astype(x.dtype)
    sin = sin.astype(x.dtype)
    xp = x.reshape(B, L, H, D // 2, 2)
    x0 = xp[..., 0]
    x1 = xp[..., 1]
    out0 = x0 * cos - x1 * sin
    out1 = x0 * sin + x1 * cos
    return mx.stack([out0, out1], axis=-1).reshape(B, L, H, D)


# --------------------------------------------------------------------------- #
# Norms / FFN / embeddings
# --------------------------------------------------------------------------- #
class LuminaRMSNormZero(nn.Module):
    def __init__(self, dim: int, norm_eps: float):
        super().__init__()
        self.linear = nn.Linear(min(dim, 1024), 4 * dim, bias=True)
        self.norm = nn.RMSNorm(dim, eps=norm_eps)

    def __call__(self, x: mx.array, temb: mx.array):
        emb = self.linear(_silu(temb))                       # [B, 4*dim]
        scale_msa, gate_msa, scale_mlp, gate_mlp = mx.split(emb, 4, axis=-1)
        x = self.norm(x) * (1 + scale_msa[:, None])
        return x, gate_msa, scale_mlp, gate_mlp


class LuminaLayerNormContinuous(nn.Module):
    """elementwise_affine=False LayerNorm + AdaLN scale + output projection."""

    def __init__(self, dim: int, cond_dim: int, out_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.linear_1 = nn.Linear(cond_dim, dim, bias=True)
        self.linear_2 = nn.Linear(dim, out_dim, bias=True)

    def __call__(self, x: mx.array, cond: mx.array) -> mx.array:
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        x = (x - mean) * mx.rsqrt(var + self.eps)
        scale = self.linear_1(_silu(cond))
        x = x * (1 + scale)[:, None, :]
        return self.linear_2(x)


class LuminaFeedForward(nn.Module):
    def __init__(self, dim: int, inner_dim: int, multiple_of: int = 256):
        super().__init__()
        inner_dim = multiple_of * ((inner_dim + multiple_of - 1) // multiple_of)
        self.linear_1 = nn.Linear(dim, inner_dim, bias=False)
        self.linear_2 = nn.Linear(inner_dim, dim, bias=False)
        self.linear_3 = nn.Linear(dim, inner_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_2(_silu(self.linear_1(x)) * self.linear_3(x))


def get_timestep_embedding(timesteps: mx.array, dim: int, scale: float = 1.0,
                           max_period: int = 10000) -> mx.array:
    """diffusers Timesteps: flip_sin_to_cos=True, downscale_freq_shift=0.0."""
    half = dim // 2
    exponent = -math.log(max_period) * mx.arange(half, dtype=mx.float32) / half
    emb = mx.exp(exponent)
    emb = (timesteps[:, None].astype(mx.float32) * emb[None, :]) * scale
    # flip_sin_to_cos -> [cos, sin]
    return mx.concatenate([mx.cos(emb), mx.sin(emb)], axis=-1)


class TimestepEmbedding(nn.Module):
    def __init__(self, in_dim: int, time_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, time_dim, bias=True)
        self.linear_2 = nn.Linear(time_dim, time_dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_2(_silu(self.linear_1(x)))


class Lumina2CombinedTimestepCaptionEmbedding(nn.Module):
    def __init__(self, hidden_size: int, instruction_feat_dim: int,
                 frequency_embedding_size: int = 256, norm_eps: float = 1e-5,
                 timestep_scale: float = 1000.0):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.timestep_scale = timestep_scale
        self.timestep_embedder = TimestepEmbedding(frequency_embedding_size, min(hidden_size, 1024))
        # Sequential(RMSNorm(instruction_feat_dim), Linear(instruction_feat_dim, hidden_size))
        self.caption_embedder = [nn.RMSNorm(instruction_feat_dim, eps=norm_eps),
                                 nn.Linear(instruction_feat_dim, hidden_size, bias=True)]

    def __call__(self, timestep: mx.array, caption: mx.array):
        t_proj = get_timestep_embedding(timestep, self.frequency_embedding_size, scale=self.timestep_scale)
        temb = self.timestep_embedder(t_proj.astype(caption.dtype))
        cap = self.caption_embedder[1](self.caption_embedder[0](caption))
        return temb, cap


# --------------------------------------------------------------------------- #
# Attention
# --------------------------------------------------------------------------- #
class Attention(nn.Module):
    """Self-attention with GQA, per-head RMSNorm q/k, 3-axis RoPE."""

    def __init__(self, dim: int, heads: int, kv_heads: int, eps: float = 1e-5):
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.to_q = nn.Linear(dim, heads * self.head_dim, bias=False)
        self.to_k = nn.Linear(dim, kv_heads * self.head_dim, bias=False)
        self.to_v = nn.Linear(dim, kv_heads * self.head_dim, bias=False)
        self.norm_q = nn.RMSNorm(self.head_dim, eps=eps)
        self.norm_k = nn.RMSNorm(self.head_dim, eps=eps)
        self.to_out = [nn.Linear(heads * self.head_dim, dim, bias=False)]

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        B, L, _ = x.shape
        q = self.norm_q(self.to_q(x).reshape(B, L, self.heads, self.head_dim))
        k = self.norm_k(self.to_k(x).reshape(B, L, self.kv_heads, self.head_dim))
        v = self.to_v(x).reshape(B, L, self.kv_heads, self.head_dim)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        out = _sdpa(q, k, v, self.heads, self.kv_heads, self.scale)
        return self.to_out[0](out)


class DoubleStreamProcessor(nn.Module):
    """Separate img/instruct q/k/v + per-stream output projections."""

    def __init__(self, dim: int, heads: int, kv_heads: int):
        super().__init__()
        head_dim = dim // heads
        self.img_to_q = nn.Linear(dim, heads * head_dim, bias=False)
        self.img_to_k = nn.Linear(dim, kv_heads * head_dim, bias=False)
        self.img_to_v = nn.Linear(dim, kv_heads * head_dim, bias=False)
        self.instruct_to_q = nn.Linear(dim, heads * head_dim, bias=False)
        self.instruct_to_k = nn.Linear(dim, kv_heads * head_dim, bias=False)
        self.instruct_to_v = nn.Linear(dim, kv_heads * head_dim, bias=False)
        self.img_out = nn.Linear(heads * head_dim, dim, bias=False)
        self.instruct_out = nn.Linear(heads * head_dim, dim, bias=False)


class DoubleStreamJointAttention(nn.Module):
    """img<->instruct joint attention over concatenated [instruct ; img]."""

    def __init__(self, dim: int, heads: int, kv_heads: int, eps: float = 1e-5):
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.norm_q = nn.RMSNorm(self.head_dim, eps=eps)
        self.norm_k = nn.RMSNorm(self.head_dim, eps=eps)
        self.to_out = [nn.Linear(heads * self.head_dim, dim, bias=False)]
        self.processor = DoubleStreamProcessor(dim, heads, kv_heads)

    def __call__(self, img: mx.array, instruct: mx.array, cos: mx.array, sin: mx.array):
        p = self.processor
        B, L_i, _ = instruct.shape
        L_img = img.shape[1]
        # concat order: instruct first, then img
        q = mx.concatenate([p.instruct_to_q(instruct), p.img_to_q(img)], axis=1)
        k = mx.concatenate([p.instruct_to_k(instruct), p.img_to_k(img)], axis=1)
        v = mx.concatenate([p.instruct_to_v(instruct), p.img_to_v(img)], axis=1)
        L = L_i + L_img
        q = self.norm_q(q.reshape(B, L, self.heads, self.head_dim))
        k = self.norm_k(k.reshape(B, L, self.kv_heads, self.head_dim))
        v = v.reshape(B, L, self.kv_heads, self.head_dim)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        out = _sdpa(q, k, v, self.heads, self.kv_heads, self.scale)   # [B, L, dim]
        instruct_out = p.instruct_out(out[:, :L_i])
        img_out = p.img_out(out[:, L_i:])
        merged = mx.concatenate([instruct_out, img_out], axis=1)
        return self.to_out[0](merged), L_i


def _sdpa(q, k, v, heads, kv_heads, scale):
    """q:[B,L,H,d] k,v:[B,L,kvH,d] -> [B,L,H*d]. GQA expand, no mask (batch=1)."""
    B, L, _, d = q.shape
    q = q.transpose(0, 2, 1, 3)                       # [B,H,L,d]
    k = k.transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)
    if heads != kv_heads:
        rep = heads // kv_heads
        k = mx.repeat(k, rep, axis=1)
        v = mx.repeat(v, rep, axis=1)
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
    return out.transpose(0, 2, 1, 3).reshape(B, L, heads * d)


# --------------------------------------------------------------------------- #
# Blocks
# --------------------------------------------------------------------------- #
class BasicBlock(nn.Module):
    """single_stream / noise_refiner (modulation) and context_refiner (no mod)."""

    def __init__(self, dim, heads, kv_heads, multiple_of, norm_eps, modulation: bool):
        super().__init__()
        self.modulation = modulation
        self.attn = Attention(dim, heads, kv_heads, eps=1e-5)
        self.feed_forward = LuminaFeedForward(dim, 4 * dim, multiple_of)
        if modulation:
            self.norm1 = LuminaRMSNormZero(dim, norm_eps)
        else:
            self.norm1 = nn.RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = nn.RMSNorm(dim, eps=norm_eps)
        self.norm2 = nn.RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = nn.RMSNorm(dim, eps=norm_eps)

    def __call__(self, x, cos, sin, temb=None):
        if self.modulation:
            xn, gate_msa, scale_mlp, gate_mlp = self.norm1(x, temb)
            attn = self.attn(xn, cos, sin)
            x = x + mx.tanh(gate_msa[:, None]) * self.norm2(attn)
            mlp = self.feed_forward(self.ffn_norm1(x) * (1 + scale_mlp[:, None]))
            x = x + mx.tanh(gate_mlp[:, None]) * self.ffn_norm2(mlp)
        else:
            attn = self.attn(self.norm1(x), cos, sin)
            x = x + self.norm2(attn)
            mlp = self.feed_forward(self.ffn_norm1(x))
            x = x + self.ffn_norm2(mlp)
        return x


class DoubleStreamBlock(nn.Module):
    def __init__(self, dim, heads, kv_heads, multiple_of, norm_eps):
        super().__init__()
        self.img_instruct_attn = DoubleStreamJointAttention(dim, heads, kv_heads, eps=1e-5)
        self.img_self_attn = Attention(dim, heads, kv_heads, eps=1e-5)
        self.img_feed_forward = LuminaFeedForward(dim, 4 * dim, multiple_of)
        self.img_norm1 = LuminaRMSNormZero(dim, norm_eps)
        self.img_norm2 = LuminaRMSNormZero(dim, norm_eps)
        self.img_norm3 = LuminaRMSNormZero(dim, norm_eps)
        self.img_ffn_norm1 = nn.RMSNorm(dim, eps=norm_eps)
        self.img_attn_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.img_self_attn_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.img_ffn_norm2 = nn.RMSNorm(dim, eps=norm_eps)
        self.instruct_feed_forward = LuminaFeedForward(dim, 4 * dim, multiple_of)
        self.instruct_norm1 = LuminaRMSNormZero(dim, norm_eps)
        self.instruct_norm2 = LuminaRMSNormZero(dim, norm_eps)
        self.instruct_ffn_norm1 = nn.RMSNorm(dim, eps=norm_eps)
        self.instruct_attn_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.instruct_ffn_norm2 = nn.RMSNorm(dim, eps=norm_eps)

    def __call__(self, img, instruct, full_cos, full_sin, img_cos, img_sin, temb):
        img_n1, img_gate_msa, img_scale_mlp, img_gate_mlp = self.img_norm1(img, temb)
        img_n2, img_shift_mlp, _, _ = self.img_norm2(img, temb)
        img_n3, img_gate_self, _, _ = self.img_norm3(img, temb)
        ins_n1, ins_gate_msa, ins_scale_mlp, ins_gate_mlp = self.instruct_norm1(instruct, temb)
        ins_n2, ins_shift_mlp, _, _ = self.instruct_norm2(instruct, temb)

        joint, L_i = self.img_instruct_attn(img_n1, ins_n1, full_cos, full_sin)
        ins_attn = joint[:, :L_i]
        img_attn = joint[:, L_i:]
        img_self = self.img_self_attn(img_n3, img_cos, img_sin)

        img = img + mx.tanh(img_gate_msa[:, None]) * self.img_attn_norm(img_attn)
        img = img + mx.tanh(img_gate_self[:, None]) * self.img_self_attn_norm(img_self)
        img_mlp_in = (1 + img_scale_mlp[:, None]) * img_n2 + img_shift_mlp[:, None]
        img_mlp = self.img_feed_forward(self.img_ffn_norm1(img_mlp_in))
        img = img + mx.tanh(img_gate_mlp[:, None]) * self.img_ffn_norm2(img_mlp)

        instruct = instruct + mx.tanh(ins_gate_msa[:, None]) * self.instruct_attn_norm(ins_attn)
        ins_mlp_in = (1 + ins_scale_mlp[:, None]) * ins_n2 + ins_shift_mlp[:, None]
        ins_mlp = self.instruct_feed_forward(self.instruct_ffn_norm1(ins_mlp_in))
        instruct = instruct + mx.tanh(ins_gate_mlp[:, None]) * self.instruct_ffn_norm2(ins_mlp)
        return img, instruct


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class BooguImageTransformer2DModel(nn.Module):
    def __init__(self, patch_size=2, in_channels=16, out_channels=None, hidden_size=3360,
                 num_layers=40, num_double_stream_layers=8, num_refiner_layers=2,
                 num_attention_heads=28, num_kv_heads=7, multiple_of=256, norm_eps=1e-5,
                 axes_dim_rope=(40, 40, 40), axes_lens=(2048, 1664, 1664),
                 instruction_feat_dim=4096, timestep_scale=1000.0, theta=10000):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.hidden_size = hidden_size
        self.axes_dim_rope = tuple(axes_dim_rope)
        self.theta = theta
        h, kv = num_attention_heads, num_kv_heads

        self.x_embedder = nn.Linear(patch_size * patch_size * in_channels, hidden_size, bias=True)
        self.ref_image_patch_embedder = nn.Linear(patch_size * patch_size * in_channels, hidden_size, bias=True)
        self.time_caption_embed = Lumina2CombinedTimestepCaptionEmbedding(
            hidden_size, instruction_feat_dim, norm_eps=norm_eps, timestep_scale=timestep_scale)

        self.noise_refiner = [BasicBlock(hidden_size, h, kv, multiple_of, norm_eps, True)
                              for _ in range(num_refiner_layers)]
        self.ref_image_refiner = [BasicBlock(hidden_size, h, kv, multiple_of, norm_eps, True)
                                  for _ in range(num_refiner_layers)]
        self.context_refiner = [BasicBlock(hidden_size, h, kv, multiple_of, norm_eps, False)
                                for _ in range(num_refiner_layers)]
        self.double_stream_layers = [DoubleStreamBlock(hidden_size, h, kv, multiple_of, norm_eps)
                                     for _ in range(num_double_stream_layers)]
        self.single_stream_layers = [BasicBlock(hidden_size, h, kv, multiple_of, norm_eps, True)
                                     for _ in range(num_layers - num_double_stream_layers)]
        self.norm_out = LuminaLayerNormContinuous(
            hidden_size, cond_dim=min(hidden_size, 1024),
            out_dim=patch_size * patch_size * self.out_channels, eps=1e-6)
        self.image_index_embedding = mx.zeros((5, hidden_size))

    @classmethod
    def from_config(cls, cfg: dict) -> "BooguImageTransformer2DModel":
        ic = cfg["instruction_feature_configs"]
        return cls(
            patch_size=cfg["patch_size"], in_channels=cfg["in_channels"],
            out_channels=cfg.get("out_channels"), hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"], num_double_stream_layers=cfg["num_double_stream_layers"],
            num_refiner_layers=cfg["num_refiner_layers"], num_attention_heads=cfg["num_attention_heads"],
            num_kv_heads=cfg["num_kv_heads"], multiple_of=cfg["multiple_of"], norm_eps=cfg["norm_eps"],
            axes_dim_rope=cfg["axes_dim_rope"], axes_lens=cfg["axes_lens"],
            instruction_feat_dim=ic["instruction_feat_dim"], timestep_scale=cfg["timestep_scale"])

    def _patchify(self, latent: mx.array) -> mx.array:
        """latent [B,C,H,W] -> tokens [B, h*w, p*p*C] in (p1 p2 c) order."""
        B, C, H, W = latent.shape
        p = self.patch_size
        ht, wt = H // p, W // p
        x = latent.reshape(B, C, ht, p, wt, p)        # B c h p1 w p2
        x = x.transpose(0, 2, 4, 3, 5, 1)             # B h w p1 p2 c
        return x.reshape(B, ht * wt, p * p * C)

    def _unpatchify(self, tokens: mx.array, H: int, W: int) -> mx.array:
        B = tokens.shape[0]
        p, C = self.patch_size, self.out_channels
        ht, wt = H // p, W // p
        x = tokens.reshape(B, ht, wt, p, p, C)        # B h w p1 p2 c
        x = x.transpose(0, 5, 1, 3, 2, 4)             # B c h p1 w p2
        return x.reshape(B, C, ht * p, wt * p)

    def _position_ids(self, L_cap: int, ht: int, wt: int) -> np.ndarray:
        cap = np.tile(np.arange(L_cap, dtype=np.int64)[:, None], (1, 3))     # [L_cap,3]
        rows = np.repeat(np.arange(ht), wt)
        cols = np.tile(np.arange(wt), ht)
        img = np.stack([np.full(ht * wt, L_cap, dtype=np.int64), rows, cols], axis=1)
        return np.concatenate([cap, img], axis=0)                            # [L_cap+L_img, 3]

    def __call__(self, latent: mx.array, timestep: mx.array,
                 instruction_hidden_states: mx.array) -> mx.array:
        """latent [1,16,H,W]; timestep [1]; instruction [1,L_cap,4096] -> [1,16,H,W]."""
        B, C, H, W = latent.shape
        p = self.patch_size
        ht, wt = H // p, W // p
        L_cap = instruction_hidden_states.shape[1]

        temb, caption = self.time_caption_embed(timestep, instruction_hidden_states)
        x = self.x_embedder(self._patchify(latent))            # [1, L_img, hidden]

        pos = self._position_ids(L_cap, ht, wt)
        cos_np, sin_np = rope_cos_sin(pos, self.axes_dim_rope, self.theta)
        full_cos = mx.array(cos_np)[None, :, None, :]          # [1, L, 1, 60]
        full_sin = mx.array(sin_np)[None, :, None, :]
        cap_cos, cap_sin = full_cos[:, :L_cap], full_sin[:, :L_cap]
        img_cos, img_sin = full_cos[:, L_cap:], full_sin[:, L_cap:]

        for layer in self.context_refiner:
            caption = layer(caption, cap_cos, cap_sin)
        for layer in self.noise_refiner:
            x = layer(x, img_cos, img_sin, temb)

        instruct = caption
        img = x
        for layer in self.double_stream_layers:
            img, instruct = layer(img, instruct, full_cos, full_sin, img_cos, img_sin, temb)

        hidden = mx.concatenate([instruct, img], axis=1)       # fuse
        for layer in self.single_stream_layers:
            hidden = layer(hidden, full_cos, full_sin, temb)

        hidden = self.norm_out(hidden, temb)
        img_tokens = hidden[:, L_cap:]                          # [1, L_img, 64]
        return self._unpatchify(img_tokens, H, W)
