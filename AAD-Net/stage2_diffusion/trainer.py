"""Stage 2 - Latent Diffusion Trainer.

Implements the standard diffusion loss (noise prediction) for training
the Cross-Lateral Diffusion U-Net. Training uses Normal patients (label=0) only.
"""

import torch
import torch.nn as nn


class LatentDiffusionTrainer:
    def __init__(self, T: int = 1000, device: str = "cuda"):
        self.T = T
        self.device = device
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.alphas = alphas.to(device)
        self.alphas_cumprod = alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = (alphas_cumprod ** 0.5).to(device)
        self.sqrt_one_minus_alphas_cumprod = ((1.0 - alphas_cumprod) ** 0.5).to(device)

    def train_step(
        self,
        l_cc: torch.Tensor,
        r_cc: torch.Tensor,
        autoencoder,
        diff_unet: nn.Module,
        timesteps: torch.Tensor = None,
    ) -> torch.Tensor:
        B = l_cc.size(0)
        if timesteps is None:
            timesteps = torch.randint(0, self.T, (B,), device=self.device).long()

        with torch.no_grad():
            z_l = autoencoder.encode(l_cc)
            z_r = autoencoder.encode(r_cc)

        noise = torch.randn_like(z_r)

        sqrt_alpha_t = self.sqrt_alphas_cumprod[timesteps].view(B, 1, 1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[timesteps].view(B, 1, 1, 1)
        z_r_noisy = sqrt_alpha_t * z_r + sqrt_one_minus_alpha_t * noise

        pred_noise = diff_unet(z_r_noisy, timesteps, condition=z_l)

        loss = nn.functional.mse_loss(pred_noise, noise)
        return loss
