# src/dataloader_stress_test.py
# Iterates through the entire training DataLoader once
# Confirms no broken images crash mid-epoch

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tqdm import tqdm
from src.dataset import get_dataloaders

train_loader, val_loader, _, _ = get_dataloaders(
    'data/processed', img_size=128, batch_size=32
)

print("Running full epoch stress test on train loader...")
total_imgs = 0
for batch_idx, (imgs, labels) in enumerate(tqdm(train_loader)):
    assert imgs.shape[1:] == torch.Size([3, 128, 128]), \
        f"Unexpected shape at batch {batch_idx}: {imgs.shape}"
    assert imgs.min() >= -1.1 and imgs.max() <= 1.1, \
        f"Pixel range out of bounds at batch {batch_idx}"
    total_imgs += imgs.shape[0]

print(f"\nStress test passed — {total_imgs} images loaded without errors")
print(f"Batches: {len(train_loader)} train, {len(val_loader)} val")
print("Ready for Phase 3 — CVAE training")