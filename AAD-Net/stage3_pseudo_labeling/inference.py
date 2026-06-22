"""Stage 3 - Bi-Directional DDIM Sampler.

Uses the trained Cross-Lateral Diffusion U-Net to generate simulated
contralateral breasts in both directions:
  Pathway A: L -> R  (hat_I_R)
  Pathway B: R -> L  (hat_I_L)
"""

import math
import torch
import torch.nn as nn


def _cosine_alphas_cumprod(T: int, s: float = 0.008) -> torch.Tensor:
    steps = torch.arange(T + 1)
    f_t = torch.cos(((steps / T + s) / (1 + s)) * math.pi * 0.5) ** 2
    return f_t / f_t[0]


def _build_schedule(T: int, num_steps: int):
    steps = torch.linspace(0, T - 1, num_steps).long()
    alpha_bar = _cosine_alphas_cumprod(T)
    betas = torch.clip(1.0 - alpha_bar[1:] / alpha_bar[:-1], 0.0001, 0.02)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return steps, alphas.to("cpu"), alphas_cumprod.to("cpu"), betas.to("cpu")


def _ddim_step(
    z_t: torch.Tensor,
    t: int,
    next_t: int,
    diff_unet: nn.Module,
    condition: torch.Tensor,
    alphas: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    device: torch.device,
):
    alpha_t = alphas[t]
    alpha_bar_t = alphas_cumprod[t]
    alpha_bar_t_next = alphas_cumprod[next_t] if next_t >= 0 else torch.tensor(1.0, device=device)

    t_tensor = torch.tensor([t], device=device, dtype=torch.long)
    with torch.no_grad():
        pred_noise = diff_unet(z_t, t_tensor, condition)

    pred_x0 = (z_t - (1 - alpha_bar_t).sqrt().to(device) * pred_noise) / alpha_bar_t.sqrt().to(device).clamp(min=1e-8)
    pred_x0 = torch.clamp(pred_x0, -5.0, 5.0)

    direction = (1 - alpha_bar_t_next.to(device)).sqrt() * pred_noise
    z_next = alpha_bar_t_next.to(device).sqrt() * pred_x0 + direction
    return z_next


def _ddim_reverse_sample(
    diff_unet: nn.Module,
    condition: torch.Tensor,
    num_steps: int = 20,
    T: int = 1000,
    latent_ch: int = 4,
    latent_h: int = 64,
    latent_w: int = 128,
    device: str = "cuda",
):
    steps, alphas, alphas_cumprod, _ = _build_schedule(T, num_steps)

    z_t = torch.randn(1, latent_ch, latent_h, latent_w, device=device)

    for i in range(len(steps) - 1, 0, -1):
        t = steps[i].item()
        next_t = steps[i - 1].item()
        z_t = _ddim_step(z_t, t, next_t, diff_unet, condition, alphas, alphas_cumprod, z_t.device)

    return z_t


def run_bilateral_inference(
    diff_unet: nn.Module,
    autoencoder,
    l_cc: torch.Tensor,
    r_cc: torch.Tensor,
    l_mlo: torch.Tensor,
    r_mlo: torch.Tensor,
    steps: int = 20,
    T: int = 1000,
):
    autoencoder.eval()
    diff_unet.eval()
    device = l_cc.device

    with torch.no_grad():
        z_l_cc = autoencoder.encode(l_cc)
        z_r_cc = autoencoder.encode(r_cc)

        z_hat_r_cc = _ddim_reverse_sample(
            diff_unet, z_l_cc,
            num_steps=steps, T=T,
            latent_ch=z_r_cc.size(1),
            latent_h=z_r_cc.size(2),
            latent_w=z_r_cc.size(3),
            device=str(device),
        )
        z_hat_l_cc = _ddim_reverse_sample(
            diff_unet, z_r_cc,
            num_steps=steps, T=T,
            latent_ch=z_l_cc.size(1),
            latent_h=z_l_cc.size(2),
            latent_w=z_l_cc.size(3),
            device=str(device),
        )

        hat_I_R = autoencoder.decode(z_hat_r_cc)
        hat_I_L = autoencoder.decode(z_hat_l_cc)

        z_l_mlo = autoencoder.encode(l_mlo)
        z_r_mlo = autoencoder.encode(r_mlo)

        z_hat_r_mlo = _ddim_reverse_sample(
            diff_unet, z_l_mlo,
            num_steps=steps, T=T,
            latent_ch=z_r_mlo.size(1),
            latent_h=z_r_mlo.size(2),
            latent_w=z_r_mlo.size(3),
            device=str(device),
        )
        z_hat_l_mlo = _ddim_reverse_sample(
            diff_unet, z_r_mlo,
            num_steps=steps, T=T,
            latent_ch=z_l_mlo.size(1),
            latent_h=z_l_mlo.size(2),
            latent_w=z_l_mlo.size(3),
            device=str(device),
        )

        hat_I_R_mlo = autoencoder.decode(z_hat_r_mlo)
        hat_I_L_mlo = autoencoder.decode(z_hat_l_mlo)

    return hat_I_R, hat_I_L, hat_I_R_mlo, hat_I_L_mlo
