# src/verify_dataset.py
# Run this to confirm the DataLoader is working before building the CVAE

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib
matplotlib.use('Agg')   # no display needed on server
import matplotlib.pyplot as plt
from src.dataset import get_dataloaders, IDX_TO_SEVERITY

DATA_DIR  = 'data/processed'
IMG_SIZE  = 128
BATCH     = 32

train_loader, val_loader, train_ds, val_ds = get_dataloaders(
    DATA_DIR, IMG_SIZE, BATCH
)

# ── Batch shape check ────────────────────────────────────────────────────────
imgs, labels = next(iter(train_loader))
print(f"\nBatch image shape : {imgs.shape}")       # expect [32, 3, 128, 128]
print(f"Batch label shape : {labels.shape}")       # expect [32]
print(f"Pixel range       : [{imgs.min():.2f}, {imgs.max():.2f}]")  # expect [-1, 1]
print(f"Labels in batch   : {sorted(labels.unique().tolist())}")

# ── Verify weighted sampler is balancing classes ─────────────────────────────
from collections import Counter
all_labels = []
for _, lbls in train_loader:
    all_labels.extend(lbls.tolist())
counts = Counter(all_labels)
print(f"\nClass balance over full epoch (weighted sampler):")
for idx in sorted(counts):
    print(f"  {IDX_TO_SEVERITY[idx]:>10} (label {idx}): {counts[idx]} samples")

# ── Save a sample grid ────────────────────────────────────────────────────────
imgs, labels = next(iter(val_loader))
fig, axes = plt.subplots(3, 6, figsize=(14, 7))
fig.suptitle('Sample images per severity class (val set, unnormalised)', fontsize=12)

shown = {0: 0, 1: 0, 2: 0}
plotted = 0

for i in range(len(labels)):
    sev = labels[i].item()
    col = shown[sev]
    if col >= 6:
        continue
    img = imgs[i].permute(1, 2, 0).numpy()
    img = (img * 0.5 + 0.5).clip(0, 1)   # [-1,1] → [0,1]
    axes[sev][col].imshow(img)
    axes[sev][col].set_title(IDX_TO_SEVERITY[sev], fontsize=8)
    axes[sev][col].axis('off')
    shown[sev] += 1
    if all(v >= 6 for v in shown.values()):
        break

plt.tight_layout()
os.makedirs('outputs', exist_ok=True)
plt.savefig('outputs/sample_grid.png', dpi=120)
print("\nSample grid saved → outputs/sample_grid.png")