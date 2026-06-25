"""Stage 2 - Cross-Lateral Diffusion U-Net with Cross-Attention layers.

This U-Net learns to generate the contralateral breast latent from the
condition latent of the reference breast. Each downsample block contains
a Cross-Attention layer where:
  - Query: output of the current block (noisy side)
  - Key/Value: projected from the condition latent
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class CrossAttention(nn.Module):
    def __init__(self, query_channels: int, cond_channels: int, heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head
        self.to_q = nn.Linear(query_channels, inner_dim, bias=False)
        self.to_k = nn.Linear(cond_channels, inner_dim, bias=False)
        self.to_v = nn.Linear(cond_channels, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_channels)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        cond_b, cond_c = condition.shape[:2]
        cond_flat = condition.reshape(cond_b, cond_c, -1).permute(0, 2, 1) if condition.dim() == 4 else condition
        q = self.to_q(x_flat)
        k = self.to_k(cond_flat)
        v = self.to_v(cond_flat)
        q = q.reshape(b, self.heads, h * w, self.dim_head)
        k = k.reshape(b, self.heads, -1, self.dim_head)
        v = v.reshape(b, self.heads, -1, self.dim_head)
        scale = self.dim_head ** -0.5
        attn = torch.einsum("bhnd,bhkd->bhnk", q, k) * scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bhnk,bhkd->bhnd", attn, v)
        out = out.reshape(b, h * w, self.heads * self.dim_head)
        out = self.to_out(out)
        out = out.reshape(b, h, w, c).permute(0, 3, 1, 2)
        return out


class ResBlock(nn.Module):
    def __init__(self, channels: int, emb_channels: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.time_emb = nn.Sequential(
            nn.Linear(emb_channels, channels),
            nn.SiLU(),
        )
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(F.silu(self.conv1(x)))
        h = h + self.time_emb(t_emb)[:, :, None, None]
        h = self.norm2(F.silu(self.conv2(h)))
        return x + h


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_ch: int, time_emb_dim: int = 256):
        super().__init__()
        self.res = ResBlock(in_ch, time_emb_dim)
        self.cross_attn = CrossAttention(in_ch, cond_ch)
        self.down = nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, cond: torch.Tensor) -> tuple:
        x = self.res(x, t_emb)
        x = x + self.cross_attn(x, cond)
        skip = x
        x = self.down(x)
        return x, skip


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_ch: int, time_emb_dim: int = 256):
        super().__init__()
        self.res = ResBlock(in_ch, time_emb_dim)
        self.cross_attn = CrossAttention(in_ch, cond_ch)
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.skip_proj = nn.Conv2d(in_ch, in_ch, 1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if skip.shape[2:] != x.shape[2:]:
            skip = F.interpolate(skip, size=x.shape[2:], mode="bilinear", align_corners=False)
        skip = self.skip_proj(skip)
        x = x + skip
        x = self.res(x, t_emb)
        x = x + self.cross_attn(x, cond)
        x = self.up(x)
        return x


class CrossLateralDiffusionUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        base_channels: int = 128,
        time_emb_dim: int = 256,
        latent_h: int = 64,
        latent_w: int = 128,
    ):
        super().__init__()
        self.latent_h = latent_h
        self.latent_w = latent_w

        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        self.first = nn.Conv2d(in_channels + in_channels, base_channels, 3, padding=1)

        self.down1 = DownBlock(base_channels, base_channels, in_channels, time_emb_dim)
        self.down2 = DownBlock(base_channels, base_channels * 2, in_channels, time_emb_dim)
        self.down3 = DownBlock(base_channels * 2, base_channels * 2, in_channels, time_emb_dim)
        self.down4 = DownBlock(base_channels * 2, base_channels * 2, in_channels, time_emb_dim)

        self.bottleneck = ResBlock(base_channels * 2, time_emb_dim)

        self.up1 = UpBlock(base_channels * 2, base_channels * 2, in_channels, time_emb_dim)
        self.up2 = UpBlock(base_channels * 2, base_channels, in_channels, time_emb_dim)
        self.up3 = UpBlock(base_channels, base_channels, in_channels, time_emb_dim)
        self.up4 = UpBlock(base_channels, base_channels, in_channels, time_emb_dim)

        final_channels = base_channels
        self.final = nn.Sequential(
            nn.GroupNorm(8, final_channels),
            nn.SiLU(),
            nn.Conv2d(final_channels, out_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(t)

        if condition.dim() == 3:
            condition = condition.unsqueeze(0)

        x = torch.cat([x, condition], dim=1)
        x = self.first(x)

        s1 = x
        x, _ = self.down1(x, t_emb, condition)
        s2 = x
        x, _ = self.down2(x, t_emb, condition)
        s3 = x
        x, _ = self.down3(x, t_emb, condition)
        s4 = x
        x, _ = self.down4(x, t_emb, condition)

        x = self.bottleneck(x, t_emb)

        x = self.up1(x, s4, t_emb, condition)
        x = self.up2(x, s3, t_emb, condition)
        x = self.up3(x, s2, t_emb, condition)
        x = self.up4(x, s1, t_emb, condition)

        x = self.final(x)
        return x
