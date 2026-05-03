# src/pool_sanity_check.py
# Verifies the augmented pool DataLoader works before Phase 5 training
# Also generates a real vs synthetic visual comparison grid for the report

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from collections import Counter

POOL_DIR   = 'outputs/augmented_pool'
REAL_DIR   = 'data/processed'
GATED_DIR  = 'outputs/gated'
REPORT_DIR = 'outputs/quality_gate'
SEVERITIES = ['mild', 'moderate', 'severe']
SEVERITY_MAP = {'mild': 0, 'moderate': 1, 'severe': 2}


class AugmentedPoolDataset(Dataset):
    """Dataset that loads from the augmented pool folder."""

    def __init__(self, root_dir, transform=None):
        self.samples   = []
        self.transform = transform

        for severity, idx in SEVERITY_MAP.items():
            folder = os.path.join(root_dir, severity)
            if not os.path.exists(folder):
                continue
            for fname in os.listdir(folder):
                if fname.lower().endswith(('.jpg','.jpeg','.png')):
                    self.samples.append(
                        (os.path.join(folder, fname), idx)
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


def run_sanity_check():
    print("=" * 55)
    print("Augmented pool sanity check")
    print("=" * 55)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    dataset = AugmentedPoolDataset(POOL_DIR, transform=transform)
    loader  = DataLoader(dataset, batch_size=32, shuffle=True,
                         num_workers=4, pin_memory=True)

    print(f"Total images in pool : {len(dataset)}")

    # check class balance
    labels = [s[1] for s in dataset.samples]
    counts = Counter(labels)
    idx_to_sev = {0:'mild', 1:'moderate', 2:'severe'}
    for idx in sorted(counts):
        print(f"  {idx_to_sev[idx]:>10} (label {idx}): {counts[idx]} images")

    # stress test — one full pass
    print("\nRunning full epoch stress test...")
    total = 0
    for imgs, lbls in loader:
        assert imgs.shape[1:] == torch.Size([3, 128, 128])
        assert imgs.min() >= -1.1 and imgs.max() <= 1.1
        total += imgs.shape[0]

    print(f"Stress test passed — {total} images loaded without errors")
    print(f"Batches: {len(loader)}")

    # ── Real vs Synthetic visual comparison grid ──────────────────────────────
    print("\nGenerating real vs synthetic comparison grid...")

    fig, axes = plt.subplots(6, 6, figsize=(15, 15))
    fig.suptitle('Real (top 3 rows) vs Synthetic gated (bottom 3 rows)',
                 fontsize=12, y=1.01)

    to_pil = transforms.ToPILImage()

    for row_offset, source_dir in enumerate([REAL_DIR, GATED_DIR]):
        for sev_idx, severity in enumerate(SEVERITIES):
            folder = os.path.join(source_dir, severity)
            files  = sorted([f for f in os.listdir(folder)
                            if f.lower().endswith(('.jpg','.jpeg','.png'))])[:6]

            row = row_offset * 3 + sev_idx
            for col, fname in enumerate(files[:6]):
                img = Image.open(os.path.join(folder, fname)).convert('RGB')
                img = img.resize((128, 128))
                axes[row][col].imshow(img)
                axes[row][col].axis('off')
                if col == 0:
                    source = 'Real' if row_offset == 0 else 'Synthetic'
                    axes[row][col].set_ylabel(
                        f'{source}\n{severity}', fontsize=9,
                        rotation=0, labelpad=60, va='center'
                    )

    plt.tight_layout()
    path = os.path.join(REPORT_DIR, 'real_vs_synthetic_grid.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison grid saved → {path}")

    print("\n" + "=" * 55)
    print("Phase 4 complete.")
    print("Augmented pool is verified and ready for Phase 5.")
    print("=" * 55)


if __name__ == '__main__':
    run_sanity_check()