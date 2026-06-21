"""FLUX.1 AutoencoderKL ported to MLX.

Isomorphic with diffusers `AutoencoderKL` (module/param names preserved) so the
PyTorch state_dict maps key-for-key — only Conv2d weights need the (O,I,H,W) ->
(O,H,W,I) transpose, handled at load time. Boogu-Image uses the stock FLUX.1-dev
VAE: 16 latent channels, block_out_channels [128,256,512,512], 2 resnets per
encoder block / 3 per decoder block, GroupNorm(32), SiLU, single-head spatial
attention in the mid block, no quant / post-quant convs.

All convolutions run in NHWC (MLX-native). Tensors enter/exit decode() and
encode() in NCHW (diffusers convention) and are transposed at the boundary.
"""

from __future__ import annotations

import math
from typing import List

import mlx.core as mx
import mlx.nn as nn


def _silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


class ResnetBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int = 32, eps: float = 1e-6):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels, eps=eps, pytorch_compatible=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels, eps=eps, pytorch_compatible=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.conv_shortcut = None

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        h = self.conv1(_silu(self.norm1(x)))
        h = self.conv2(_silu(self.norm2(h)))
        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)
        return residual + h


class Downsample2D(nn.Module):
    """diffusers Downsample2D: asymmetric pad (0,1,0,1) then stride-2 conv, pad=0."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=0)

    def __call__(self, x: mx.array) -> mx.array:
        # NHWC: pad bottom of H and right of W by 1.
        x = mx.pad(x, [(0, 0), (0, 1), (0, 1), (0, 0)])
        return self.conv(x)


class Upsample2D(nn.Module):
    """diffusers Upsample2D: nearest x2 then stride-1 conv, pad=1."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        b, h, w, c = x.shape
        # nearest-neighbour upsample by 2 in H and W (NHWC).
        x = mx.broadcast_to(x[:, :, None, :, None, :], (b, h, 2, w, 2, c))
        x = x.reshape(b, h * 2, w * 2, c)
        return self.conv(x)


class VAEAttention(nn.Module):
    """Single-head spatial self-attention used in the VAE mid block."""

    def __init__(self, channels: int, groups: int = 32, eps: float = 1e-6):
        super().__init__()
        self.group_norm = nn.GroupNorm(groups, channels, eps=eps, pytorch_compatible=True)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = [nn.Linear(channels, channels)]
        self.scale = 1.0 / math.sqrt(channels)

    def __call__(self, x: mx.array) -> mx.array:
        # x: NHWC
        b, h, w, c = x.shape
        residual = x
        y = self.group_norm(x)
        y = y.reshape(b, h * w, c)
        q = self.to_q(y)
        k = self.to_k(y)
        v = self.to_v(y)
        attn = mx.softmax((q @ k.transpose(0, 2, 1)) * self.scale, axis=-1)
        out = attn @ v
        out = self.to_out[0](out)
        out = out.reshape(b, h, w, c)
        return residual + out


class UNetMidBlock2D(nn.Module):
    def __init__(self, channels: int, groups: int = 32, eps: float = 1e-6):
        super().__init__()
        self.resnets = [
            ResnetBlock2D(channels, channels, groups, eps),
            ResnetBlock2D(channels, channels, groups, eps),
        ]
        self.attentions = [VAEAttention(channels, groups, eps)]

    def __call__(self, x: mx.array) -> mx.array:
        x = self.resnets[0](x)
        x = self.attentions[0](x)
        x = self.resnets[1](x)
        return x


class DownEncoderBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_layers: int,
                 add_downsample: bool, groups: int = 32, eps: float = 1e-6):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            resnets.append(ResnetBlock2D(in_channels if i == 0 else out_channels, out_channels, groups, eps))
        self.resnets = resnets
        self.downsamplers = [Downsample2D(out_channels)] if add_downsample else None

    def __call__(self, x: mx.array) -> mx.array:
        for r in self.resnets:
            x = r(x)
        if self.downsamplers is not None:
            x = self.downsamplers[0](x)
        return x


class UpDecoderBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_layers: int,
                 add_upsample: bool, groups: int = 32, eps: float = 1e-6):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            resnets.append(ResnetBlock2D(in_channels if i == 0 else out_channels, out_channels, groups, eps))
        self.resnets = resnets
        self.upsamplers = [Upsample2D(out_channels)] if add_upsample else None

    def __call__(self, x: mx.array) -> mx.array:
        for r in self.resnets:
            x = r(x)
        if self.upsamplers is not None:
            x = self.upsamplers[0](x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels: int, latent_channels: int, block_out_channels: List[int],
                 layers_per_block: int, groups: int = 32, eps: float = 1e-6, double_z: bool = True):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1)
        down_blocks = []
        output_channel = block_out_channels[0]
        for i, boc in enumerate(block_out_channels):
            input_channel = output_channel
            output_channel = boc
            is_final = i == len(block_out_channels) - 1
            down_blocks.append(DownEncoderBlock2D(
                input_channel, output_channel, layers_per_block,
                add_downsample=not is_final, groups=groups, eps=eps))
        self.down_blocks = down_blocks
        self.mid_block = UNetMidBlock2D(block_out_channels[-1], groups, eps)
        self.conv_norm_out = nn.GroupNorm(groups, block_out_channels[-1], eps=eps, pytorch_compatible=True)
        conv_out_channels = 2 * latent_channels if double_z else latent_channels
        self.conv_out = nn.Conv2d(block_out_channels[-1], conv_out_channels, kernel_size=3, stride=1, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_in(x)
        for b in self.down_blocks:
            x = b(x)
        x = self.mid_block(x)
        x = self.conv_out(_silu(self.conv_norm_out(x)))
        return x


class Decoder(nn.Module):
    def __init__(self, out_channels: int, latent_channels: int, block_out_channels: List[int],
                 layers_per_block: int, groups: int = 32, eps: float = 1e-6):
        super().__init__()
        reversed_boc = list(reversed(block_out_channels))
        self.conv_in = nn.Conv2d(latent_channels, reversed_boc[0], kernel_size=3, stride=1, padding=1)
        self.mid_block = UNetMidBlock2D(reversed_boc[0], groups, eps)
        up_blocks = []
        output_channel = reversed_boc[0]
        for i, boc in enumerate(reversed_boc):
            input_channel = output_channel
            output_channel = boc
            is_final = i == len(reversed_boc) - 1
            up_blocks.append(UpDecoderBlock2D(
                input_channel, output_channel, layers_per_block + 1,
                add_upsample=not is_final, groups=groups, eps=eps))
        self.up_blocks = up_blocks
        self.conv_norm_out = nn.GroupNorm(groups, reversed_boc[-1], eps=eps, pytorch_compatible=True)
        self.conv_out = nn.Conv2d(reversed_boc[-1], out_channels, kernel_size=3, stride=1, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_in(x)
        x = self.mid_block(x)
        for b in self.up_blocks:
            x = b(x)
        x = self.conv_out(_silu(self.conv_norm_out(x)))
        return x


class AutoencoderKL(nn.Module):
    """FLUX.1 VAE. encode()/decode() take and return NCHW (diffusers convention)."""

    def __init__(self, in_channels: int = 3, out_channels: int = 3, latent_channels: int = 16,
                 block_out_channels: List[int] = (128, 256, 512, 512), layers_per_block: int = 2,
                 norm_num_groups: int = 32, scaling_factor: float = 0.3611,
                 shift_factor: float = 0.1159, eps: float = 1e-6):
        super().__init__()
        self.latent_channels = latent_channels
        self.scaling_factor = scaling_factor
        self.shift_factor = shift_factor
        self.encoder = Encoder(in_channels, latent_channels, list(block_out_channels),
                               layers_per_block, norm_num_groups, eps, double_z=True)
        self.decoder = Decoder(out_channels, latent_channels, list(block_out_channels),
                               layers_per_block, norm_num_groups, eps)

    @classmethod
    def from_config(cls, cfg: dict) -> "AutoencoderKL":
        return cls(
            in_channels=cfg.get("in_channels", 3),
            out_channels=cfg.get("out_channels", 3),
            latent_channels=cfg.get("latent_channels", 16),
            block_out_channels=cfg.get("block_out_channels", [128, 256, 512, 512]),
            layers_per_block=cfg.get("layers_per_block", 2),
            norm_num_groups=cfg.get("norm_num_groups", 32),
            scaling_factor=cfg.get("scaling_factor", 0.3611),
            shift_factor=cfg.get("shift_factor", 0.1159),
        )

    def encode_moments(self, x_nchw: mx.array) -> mx.array:
        """Return raw moments (mean, logvar concatenated on channel), NCHW."""
        x = x_nchw.transpose(0, 2, 3, 1)
        moments = self.encoder(x)
        return moments.transpose(0, 3, 1, 2)

    def decode(self, z_nchw: mx.array) -> mx.array:
        z = z_nchw.transpose(0, 2, 3, 1)
        out = self.decoder(z)
        return out.transpose(0, 3, 1, 2)
