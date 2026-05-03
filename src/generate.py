# src/generate.py
# Generates synthetic images using the trained cGAN
# conditioned on CVAE severity cluster distributions

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.cgan    import Generator, LATENT_DIM, NUM_CLASSES
from src.dataset import IDX_TO_SEVERITY

CGAN_MODEL_PATH = 'models/cgan_best.pth'
CVAE_MODEL_PATH = 'models/cvae_best.pth'
GENERATED_DIR   = 'outputs/generated'
DATA_DIR        = 'data/processed'
N_PER_CLASS     = 500
SEVERITY_NAMES  = {0: 'mild', 1: 'moderate', 2: 'severe'}
SEED            = 42

os.makedirs(GENERATED_DIR, exist_ok=True)
for sev in SEVERITY_NAMES.values():
    os.makedirs(os.path.join(GENERATED_DIR, sev), exist_ok=True)


def load_cvae_cluster_stats():
    """Extract per-severity mu and sigma from trained CVAE."""
    from src.model   import CVAE
    from src.dataset import get_dataloaders

    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(CVAE_MODEL_PATH, map_location=device)
    cfg        = checkpoint['config']
    model      = CVAE(cfg['latent_dim'], cfg['num_classes']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    train_loader, _, _, _ = get_dataloaders(DATA_DIR, cfg['img_size'], 32)

    all_mu = {0: [], 1: [], 2: []}
    all_lv = {0: [], 1: [], 2: []}

    with torch.no_grad():
        for imgs, labels in tqdm(train_loader, desc='Extracting CVAE clusters'):
            imgs, labels = imgs.to(device), labels.to(device)
            mu, lv = model.encoder(imgs, labels)
            for s in range(3):
                m = labels == s
                if m.sum():
                    all_mu[s].append(mu[m].cpu().numpy())
                    all_lv[s].append(lv[m].cpu().numpy())

    stats = {}
    for s in range(3):
        mu_arr = np.concatenate(all_mu[s])
        lv_arr = np.concatenate(all_lv[s])
        stats[s] = {
            'mu' : mu_arr.mean(axis=0),
            'std': np.exp(0.5 * lv_arr).mean(axis=0)
        }
        print(f"  {SEVERITY_NAMES[s]:>10}: {len(mu_arr)} samples")
    return stats


def generate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # load CVAE cluster stats
    print("Step 1: Extracting CVAE severity cluster statistics...")
    cluster_stats = load_cvae_cluster_stats()

    # load trained cGAN generator
    print("\nStep 2: Loading trained cGAN generator...")
    checkpoint = torch.load(CGAN_MODEL_PATH, map_location=device)
    G = Generator(LATENT_DIM, NUM_CLASSES).to(device)
    G.load_state_dict(checkpoint['G_state'])
    G.eval()
    print(f"Loaded cGAN from epoch {checkpoint['epoch']}")

    # generate images per severity class
    print(f"\nStep 3: Generating {N_PER_CLASS} images per class...\n")
    rng   = np.random.RandomState(SEED)
    total = 0

    for sev_idx, sev_name in SEVERITY_NAMES.items():
        stats   = cluster_stats[sev_idx]
        mu      = stats['mu']
        std     = stats['std']
        out_dir = os.path.join(GENERATED_DIR, sev_name)

        print(f"Generating {sev_name} ({N_PER_CLASS} images)...")

        for i in tqdm(range(N_PER_CLASS)):
            # sample from CVAE severity distribution
            # interpolate to GAN latent dim if needed
            eps  = rng.randn(LATENT_DIM).astype(np.float32)
            if len(mu) == LATENT_DIM:
                z_np = mu + eps * std
            else:
                z_np = np.interp(
                    np.linspace(0, 1, LATENT_DIM),
                    np.linspace(0, 1, len(mu)),
                    mu + eps * std
                ).astype(np.float32)

            z     = torch.from_numpy(z_np).unsqueeze(0).to(device)
            label = torch.tensor([sev_idx], device=device)

            with torch.no_grad():
                img = G(z, label)

            img = (img.clamp(-1, 1) + 1) / 2 * 255
            img = img[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            Image.fromarray(img).save(
                os.path.join(out_dir, f'{sev_name}_{i:04d}.png')
            )
            total += 1

    print(f"\nGeneration complete: {total} images → {GENERATED_DIR}/")
    for sev_name in SEVERITY_NAMES.values():
        count = len(os.listdir(os.path.join(GENERATED_DIR, sev_name)))
        print(f"  {sev_name:>10}: {count} images")
    print("\nNext step: Phase 4 quality gate")


if __name__ == '__main__':
    generate()