# src/visualise.py
# Post-training visualisation — run after training completes
#
# Generates:
#   1. t-SNE plot of latent space (coloured by severity) → report figure
#   2. Reconstruction quality grid (all val images)
#   3. Latent space statistics (mu and sigma per severity class)
#   4. Interpolation between severity classes in latent space

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

from src.model   import CVAE
from src.dataset import get_dataloaders, IDX_TO_SEVERITY, SEVERITY_MAP

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'models/cvae_best.pth'
OUTPUT_DIR = 'outputs/visualisations'
DATA_DIR   = 'data/processed'
COLORS     = {0: '#4A90D9', 1: '#E8A838', 2: '#D94A4A'}
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    cfg        = checkpoint['config']
    model      = CVAE(cfg['latent_dim'], cfg['num_classes']).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded model from epoch {checkpoint['epoch']}, "
          f"val loss = {checkpoint['val_loss']:.5f}")
    return model, cfg


def extract_latents(model, loader):
    """Pass all images through encoder, collect mu vectors and labels."""
    all_mu     = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Extracting latents"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            mu, log_var  = model.encoder(imgs, labels)
            all_mu.append(mu.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    return np.concatenate(all_mu), np.concatenate(all_labels)


# ── 1. t-SNE latent space plot ────────────────────────────────────────────────
def plot_tsne(model, train_loader, val_loader):
    """
    Projects 128-dim latent vectors to 2D using t-SNE.
    Well-trained CVAE shows 3 distinct but overlapping clusters.
    This is one of the most important figures in the report.
    """
    print("\nExtracting latent vectors from train + val set...")
    mu_train, labels_train = extract_latents(model, train_loader)
    mu_val,   labels_val   = extract_latents(model, val_loader)

    all_mu     = np.concatenate([mu_train, mu_val])
    all_labels = np.concatenate([labels_train, labels_val])
    is_val     = np.array([False]*len(mu_train) + [True]*len(mu_val))

    print(f"Running t-SNE on {len(all_mu)} points (latent_dim=128)...")
    tsne       = TSNE(n_components=2, perplexity=30, random_state=42,
                      max_iter=1000, verbose=1)
    embedding  = tsne.fit_transform(all_mu)

    fig, ax = plt.subplots(figsize=(9, 7))
    for sev_idx, sev_name in IDX_TO_SEVERITY.items():
        mask      = (all_labels == sev_idx) & ~is_val
        mask_val  = (all_labels == sev_idx) & is_val
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=COLORS[sev_idx], label=f'{sev_name} (train)',
                   alpha=0.6, s=25, edgecolors='none')
        ax.scatter(embedding[mask_val, 0], embedding[mask_val, 1],
                   c=COLORS[sev_idx], label=f'{sev_name} (val)',
                   alpha=1.0, s=60, marker='*', edgecolors='black',
                   linewidths=0.5)

    ax.set_title('t-SNE of CVAE latent space — coloured by severity',
                 fontsize=13)
    ax.set_xlabel('t-SNE dim 1')
    ax.set_ylabel('t-SNE dim 2')
    ax.legend(fontsize=9, ncol=2)
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'tsne_latent_space.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"t-SNE plot saved → {path}")
    return embedding, all_labels


# ── 2. Latent space statistics per severity class ─────────────────────────────
def plot_latent_stats(model, train_loader):
    """
    Shows mean and std of mu vectors per severity class.
    Confirms that encoder learned distinct distributions per class.
    """
    mu_all, labels_all = extract_latents(model, train_loader)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Latent space statistics per severity class', fontsize=12)

    for sev_idx, sev_name in IDX_TO_SEVERITY.items():
        mask    = labels_all == sev_idx
        mu_sev  = mu_all[mask]
        means   = mu_sev.mean(axis=0)
        stds    = mu_sev.std(axis=0)
        dims    = np.arange(len(means))

        axes[0].plot(dims[:50], means[:50],
                     color=COLORS[sev_idx], label=sev_name, alpha=0.8)
        axes[1].plot(dims[:50], stds[:50],
                     color=COLORS[sev_idx], label=sev_name, alpha=0.8)

    axes[0].set_title('Mean of mu per latent dimension (first 50 dims)')
    axes[0].set_xlabel('Latent dimension')
    axes[0].set_ylabel('Mean value')
    axes[0].legend()
    axes[1].set_title('Std of mu per latent dimension (first 50 dims)')
    axes[1].set_xlabel('Latent dimension')
    axes[1].set_ylabel('Std value')
    axes[1].legend()

    for ax in axes:
        ax.spines[['top','right']].set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'latent_stats.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Latent stats saved → {path}")


# ── 3. Severity interpolation in latent space ─────────────────────────────────
def plot_severity_interpolation(model, train_loader):
    """
    Interpolates between mild and severe in latent space.
    Shows smooth transition: mild → moderate → severe.
    Powerful visual for the report and demo.
    """
    mu_all, labels_all = extract_latents(model, train_loader)

    # mean latent vector per severity
    mu_mild   = torch.tensor(
        mu_all[labels_all == 0].mean(axis=0), dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)
    mu_severe = torch.tensor(
        mu_all[labels_all == 2].mean(axis=0), dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    n_steps = 8
    alphas  = np.linspace(0, 1, n_steps)

    fig, axes = plt.subplots(1, n_steps, figsize=(n_steps * 2.5, 3))
    fig.suptitle('Latent space interpolation: mild → severe', fontsize=11)

    model.eval()
    with torch.no_grad():
        for i, alpha in enumerate(alphas):
            # interpolate in latent space
            z      = (1 - alpha) * mu_mild + alpha * mu_severe

            # decode with mild label (label 0) — we're changing z, not label
            label  = torch.tensor([0], device=DEVICE)
            img    = model.decode(z, label)

            img_np = img[0].cpu().numpy().transpose(1, 2, 0)
            img_np = (img_np * 0.5 + 0.5).clip(0, 1)

            axes[i].imshow(img_np)
            axes[i].set_title(f'α={alpha:.2f}', fontsize=8)
            axes[i].axis('off')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'severity_interpolation.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Interpolation plot saved → {path}")


# ── Run all visualisations ────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Loading trained model...")
    model, cfg = load_model()

    train_loader, val_loader, _, _ = get_dataloaders(
        DATA_DIR, cfg['img_size'], cfg['batch_size']
    )

    plot_tsne(model, train_loader, val_loader)
    plot_latent_stats(model, train_loader)
    plot_severity_interpolation(model, train_loader)

    print(f"\nAll visualisations saved to {OUTPUT_DIR}/")
    print("These are your key report figures for Section 4 and 5.")