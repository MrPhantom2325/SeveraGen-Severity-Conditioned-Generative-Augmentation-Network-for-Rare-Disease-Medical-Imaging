# src/train_gan.py
# Training loop for the custom conditional GAN
#
# Uses WGAN-GP loss — stable on small datasets, no mode collapse.
# Trains discriminator 5x per generator step (standard WGAN ratio).
# Saves checkpoints and image previews every N epochs.

import os, sys, time, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.cgan    import Generator, Discriminator, gradient_penalty, LATENT_DIM, NUM_CLASSES
from src.dataset import get_dataloaders, IDX_TO_SEVERITY

# ── Config ─────────────────────────────────────────────────────────────────────
CONFIG = {
    'data_dir'      : 'data/processed',
    'output_dir'    : 'outputs/cgan',
    'model_dir'     : 'models',
    'img_size'      : 128,
    'batch_size'    : 32,
    'epochs'        : 300,
    'lr_g'          : 1e-4,
    'lr_d'          : 4e-4,       # D learns faster than G (WGAN standard)
    'n_critic'      : 5,          # D steps per G step
    'lambda_gp'     : 10,         # gradient penalty weight
    'save_every'    : 50,
    'preview_every' : 25,
    'n_preview'     : 6,          # images per class in preview grid
    'latent_dim'    : LATENT_DIM,
    'num_classes'   : NUM_CLASSES,
}


def save_preview(G, epoch, cfg, device, fixed_z, fixed_labels):
    """Save a grid of generated images — 3 rows (severity) x n_preview cols."""
    G.eval()
    with torch.no_grad():
        fake = G(fixed_z, fixed_labels)   # [n_preview*3, 3, 128, 128]

    n    = cfg['n_preview']
    fig, axes = plt.subplots(3, n, figsize=(n * 2.5, 8))
    fig.suptitle(f'Epoch {epoch} — generated images per severity class', fontsize=11)

    for row in range(3):
        for col in range(n):
            idx = row * n + col
            img = fake[idx].cpu().permute(1, 2, 0).numpy()
            img = (img * 0.5 + 0.5).clip(0, 1)
            axes[row][col].imshow(img)
            axes[row][col].axis('off')
            if col == 0:
                axes[row][col].set_ylabel(IDX_TO_SEVERITY[row],
                                          fontsize=10, rotation=0,
                                          labelpad=45, va='center')

    plt.tight_layout()
    os.makedirs(f"{cfg['output_dir']}/previews", exist_ok=True)
    path = f"{cfg['output_dir']}/previews/epoch_{epoch:04d}.png"
    plt.savefig(path, dpi=100)
    plt.close()
    G.train()
    return path


def save_loss_curves(g_losses, d_losses, cfg):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(g_losses, color='#4A90D9', label='G loss')
    axes[0].set_title('Generator loss')
    axes[0].set_xlabel('Epoch')
    axes[1].plot(d_losses, color='#D94A4A', label='D loss (WGAN)')
    axes[1].set_title('Discriminator loss (Wasserstein)')
    axes[1].set_xlabel('Epoch')
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{cfg['output_dir']}/cgan_loss_curves.png", dpi=150)
    plt.close()


def train():
    cfg    = CONFIG
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice     : {device}")
    if device.type == 'cuda':
        print(f"GPU        : {torch.cuda.get_device_name(0)}")

    os.makedirs(cfg['output_dir'], exist_ok=True)
    os.makedirs(cfg['model_dir'],  exist_ok=True)

    # data
    train_loader, _, _, _ = get_dataloaders(
        cfg['data_dir'], cfg['img_size'], cfg['batch_size']
    )
    print(f"Train batches: {len(train_loader)}")

    # models
    G = Generator(cfg['latent_dim'], cfg['num_classes']).to(device)
    D = Discriminator(cfg['num_classes']).to(device)

    g_params = sum(p.numel() for p in G.parameters())
    d_params = sum(p.numel() for p in D.parameters())
    print(f"G params   : {g_params:,}")
    print(f"D params   : {d_params:,}")

    opt_G = optim.Adam(G.parameters(), lr=cfg['lr_g'], betas=(0.0, 0.9))
    opt_D = optim.Adam(D.parameters(), lr=cfg['lr_d'], betas=(0.0, 0.9))

    # fixed noise for consistent previews
    n   = cfg['n_preview']
    fixed_z = torch.randn(n * 3, cfg['latent_dim'], device=device)
    fixed_labels = torch.tensor(
        [cls for cls in range(3) for _ in range(n)], device=device
    )

    # logging
    log_path = f"{cfg['output_dir']}/cgan_training_log.csv"
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch','g_loss','d_loss','time_s'])

    g_losses, d_losses = [], []

    print(f"\nStarting training for {cfg['epochs']} epochs...")
    print(f"{'Epoch':>6} {'G_loss':>10} {'D_loss':>10} {'Time':>8}")
    print("-" * 40)

    for epoch in range(1, cfg['epochs'] + 1):
        G.train(); D.train()
        epoch_g, epoch_d = 0.0, 0.0
        g_steps = 0
        t0      = time.time()

        for batch_idx, (real_imgs, labels) in enumerate(train_loader):
            real_imgs = real_imgs.to(device)
            labels    = labels.to(device)
            B         = real_imgs.shape[0]

            # ── Train Discriminator (n_critic steps) ──────────────────────────
            for _ in range(cfg['n_critic']):
                z    = torch.randn(B, cfg['latent_dim'], device=device)
                fake = G(z, labels).detach()

                d_real = D(real_imgs, labels).mean()
                d_fake = D(fake, labels).mean()
                gp     = gradient_penalty(D, real_imgs, fake, labels, device)
                d_loss = d_fake - d_real + cfg['lambda_gp'] * gp

                opt_D.zero_grad()
                d_loss.backward()
                opt_D.step()

            epoch_d += (d_fake - d_real).item()

            # ── Train Generator ───────────────────────────────────────────────
            z    = torch.randn(B, cfg['latent_dim'], device=device)
            fake = G(z, labels)
            g_loss = -D(fake, labels).mean()

            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

            epoch_g += g_loss.item()
            g_steps += 1

        epoch_g /= g_steps
        epoch_d /= g_steps
        elapsed  = time.time() - t0

        g_losses.append(epoch_g)
        d_losses.append(epoch_d)

        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, f'{epoch_g:.4f}',
                                    f'{epoch_d:.4f}', f'{elapsed:.1f}'])

        print(f"{epoch:>6} {epoch_g:>10.4f} {epoch_d:>10.4f} {elapsed:>6.1f}s")

        # previews
        if epoch % cfg['preview_every'] == 0 or epoch == 1:
            path = save_preview(G, epoch, cfg, device, fixed_z, fixed_labels)
            print(f"  Preview → {path}")

        # checkpoints
        if epoch % cfg['save_every'] == 0:
            torch.save({
                'epoch': epoch,
                'G_state': G.state_dict(),
                'D_state': D.state_dict(),
                'config': cfg,
            }, f"{cfg['model_dir']}/cgan_epoch_{epoch:04d}.pth")

    # save final model
    torch.save({
        'epoch': cfg['epochs'],
        'G_state': G.state_dict(),
        'D_state': D.state_dict(),
        'config': cfg,
    }, f"{cfg['model_dir']}/cgan_best.pth")

    save_loss_curves(g_losses, d_losses, cfg)

    print(f"\nTraining complete.")
    print(f"Model saved  : {cfg['model_dir']}/cgan_best.pth")
    print(f"Loss curves  : {cfg['output_dir']}/cgan_loss_curves.png")
    print(f"Previews     : {cfg['output_dir']}/previews/")


if __name__ == '__main__':
    train()