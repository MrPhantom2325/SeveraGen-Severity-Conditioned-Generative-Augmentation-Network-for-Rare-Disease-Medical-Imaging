# src/loss.py
# CVAE loss function = Reconstruction Loss + beta * KL Divergence
#
# Why two terms?
#   Reconstruction loss : forces decoder to produce images similar to input
#   KL divergence       : forces latent space to be a smooth Gaussian
#                         so we can SAMPLE from it later (Phase 4)
#
# beta parameter controls the trade-off:
#   beta too high → blurry images (KL dominates, ignores image detail)
#   beta too low  → sharp images but non-sampleable latent space
#   beta = 0.001  is a good starting point for 128x128 images

import torch
import torch.nn.functional as F


def reconstruction_loss(x_recon, x_target):
    """
    Mean Squared Error between reconstructed and original image.
    Both tensors are in [-1, 1].

    MSE is used instead of BCE because:
    - Images are continuous-valued (not binary)
    - MSE better preserves colour fidelity in medical images

    Args:
        x_recon  : [B, 3, 128, 128]  decoder output
        x_target : [B, 3, 128, 128]  original input image
    Returns:
        scalar loss (mean over batch and pixels)
    """
    return F.mse_loss(x_recon, x_target, reduction='mean')


def kl_divergence_loss(mu, log_var):
    """
    KL divergence between learned distribution N(mu, sigma^2)
    and standard normal N(0, I).

    Formula (closed form):
        KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))

    Intuition:
        Penalises the encoder for creating distributions that are
        far from N(0,1). Forces each severity cluster to sit in a
        well-defined, compact region of the latent space.

    Args:
        mu      : [B, latent_dim]
        log_var : [B, latent_dim]
    Returns:
        scalar loss (mean over batch and latent dimensions)
    """
    kl = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return kl


def cvae_loss(x_recon, x_target, mu, log_var, beta=0.001):
    """
    Total CVAE loss = Reconstruction Loss + beta * KL Divergence

    Args:
        x_recon  : [B, 3, 128, 128]  decoder output
        x_target : [B, 3, 128, 128]  original input image
        mu       : [B, latent_dim]
        log_var  : [B, latent_dim]
        beta     : float  weight on KL term (default 0.001)

    Returns:
        total_loss  : scalar
        recon_loss  : scalar (for logging)
        kl_loss     : scalar (for logging)
    """
    recon_loss = reconstruction_loss(x_recon, x_target)
    kl_loss    = kl_divergence_loss(mu, log_var)
    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    B, C, H, W  = 8, 3, 128, 128
    latent_dim  = 128

    x_recon  = torch.randn(B, C, H, W)
    x_target = torch.randn(B, C, H, W)
    mu       = torch.randn(B, latent_dim)
    log_var  = torch.randn(B, latent_dim)

    total, recon, kl = cvae_loss(x_recon, x_target, mu, log_var, beta=0.001)

    print(f"Reconstruction loss : {recon.item():.4f}")
    print(f"KL divergence loss  : {kl.item():.4f}")
    print(f"Total loss          : {total.item():.4f}")
    print("loss.py sanity check passed.")