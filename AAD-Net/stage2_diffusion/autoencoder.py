"""Stage 2 - Pre-trained Autoencoder (VQ-GAN / KL-Autoencoder) loader.

Provides a frozen encoder/decoder that maps images into a compressed latent space.
A dummy implementation is provided so Stage 2 training can be debugged before
the actual pre-trained weights are downloaded.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DummyEncoder(nn.Module):
    def __init__(self, in_channels=1, latent_channels=4, latent_h=64, latent_w=128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, latent_channels, 3, stride=1, padding=1)
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.latent_channels = latent_channels

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return x


class DummyDecoder(nn.Module):
    def __init__(self, latent_channels=4, out_channels=1, out_h=512, out_w=1024):
        super().__init__()
        self.latent_channels = latent_channels
        self.out_h = out_h
        self.out_w = out_w
        self.conv1 = nn.ConvTranspose2d(latent_channels, 128, 3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.conv4 = nn.ConvTranspose2d(32, out_channels, 3, stride=1, padding=1)
        self.out_channels = out_channels

    def forward(self, z):
        z = F.relu(self.conv1(z))
        z = F.relu(self.conv2(z))
        z = F.relu(self.conv3(z))
        z = self.conv4(z)
        return z


class AutoencoderWrapper(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg
        latent_channels = cfg.latent_channels if cfg else 4
        latent_h = cfg.latent_height if cfg else 64
        latent_w = cfg.latent_width if cfg else 128
        self.encoder = DummyEncoder(latent_channels=latent_channels, latent_h=latent_h, latent_w=latent_w)
        self.decoder = DummyDecoder(latent_channels=latent_channels)
        self.latent_channels = latent_channels
        self.latent_h = latent_h
        self.latent_w = latent_w
        self._dummy = True

    def encode(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon

    def load_pretrained(self, ckpt_path: str, device: str = "cuda"):
        import torch
        try:
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
            self.load_state_dict(state, strict=False)
            self._dummy = False
            print(f"[Autoencoder] Loaded pretrained weights from {ckpt_path}")
        except Exception as e:
            print(f"[Autoencoder] Could not load {ckpt_path} ({e}). Using dummy weights.")


def load_vqgan(cfg, device="cuda"):
    model = AutoencoderWrapper(cfg)
    model.to(device)

    if cfg.autoencoder_ckpt.exists():
        model.load_pretrained(str(cfg.autoencoder_ckpt), device)

    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model
