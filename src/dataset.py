# src/dataset.py
# Custom PyTorch Dataset for severity-labelled rare disease images
# Handles loading, preprocessing, and severity-conditioned sampling

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from collections import Counter


# ── Label map ───────────────────────────────────────────────────────────────
SEVERITY_MAP = {'mild': 0, 'moderate': 1, 'severe': 2}
IDX_TO_SEVERITY = {v: k for k, v in SEVERITY_MAP.items()}


# ── Dataset class ────────────────────────────────────────────────────────────
class RareDiseaseDataset(Dataset):
    """
    Loads images from data/processed/{mild, moderate, severe}/
    Returns (image_tensor, severity_label) pairs.
    Image size: 128x128, normalised to [-1, 1].
    """

    def __init__(self, root_dir, split='train', val_split=0.15, transform=None, seed=42):
        """
        Args:
            root_dir  : path to data/processed/
            split     : 'train' or 'val'
            val_split : fraction of data held out for validation
            transform : optional extra transforms (augmentation)
            seed      : random seed for reproducible split
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []   # list of (img_path, severity_idx)

        # collect all image paths and labels
        all_samples = []
        for severity, idx in SEVERITY_MAP.items():
            folder = os.path.join(root_dir, severity)
            if not os.path.exists(folder):
                print(f"Warning: folder not found — {folder}")
                continue
            for fname in sorted(os.listdir(folder)):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_samples.append((os.path.join(folder, fname), idx))

        # reproducible train / val split
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(all_samples))
        val_size = int(len(all_samples) * val_split)

        if split == 'val':
            selected = indices[:val_size]
        else:
            selected = indices[val_size:]

        self.samples = [all_samples[i] for i in selected]

        # class counts for reporting
        labels = [s[1] for s in self.samples]
        self.class_counts = Counter(labels)
        print(f"[{split}] loaded {len(self.samples)} images — "
              f"mild: {self.class_counts[0]}, "
              f"moderate: {self.class_counts[1]}, "
              f"severe: {self.class_counts[2]}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, severity_idx = self.samples[idx]

        # load and convert to RGB (handles greyscale or RGBA edge cases)
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(severity_idx, dtype=torch.long)


# ── Transforms ───────────────────────────────────────────────────────────────
def get_transforms(split='train', img_size=128):
    """
    Train : resize → random flip → slight rotation → tensor → normalise [-1,1]
    Val   : resize → tensor → normalise [-1,1]
    Normalisation to [-1,1] is standard for VAE/GAN training.
    """
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),                          # [0,1]
            transforms.Normalize([0.5, 0.5, 0.5],          # → [-1,1]
                                  [0.5, 0.5, 0.5])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                  [0.5, 0.5, 0.5])
        ])


# ── Weighted sampler (fixes class imbalance) ──────────────────────────────────
def get_weighted_sampler(dataset):
    """
    Gives each sample a weight inversely proportional to its class size.
    Mild (354) gets downsampled, severe (30) gets upsampled — all classes
    seen at equal frequency during training.
    """
    labels = [s[1] for s in dataset.samples]
    class_counts = Counter(labels)
    total = sum(class_counts.values())

    # weight per class = total / count  (rare class gets higher weight)
    class_weights = {cls: total / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in labels]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


# ── DataLoader factory ────────────────────────────────────────────────────────
def get_dataloaders(data_dir, img_size=128, batch_size=32):
    """
    Returns train_loader, val_loader, and both dataset objects.
    Train loader uses weighted sampling to balance severity classes.
    """
    train_dataset = RareDiseaseDataset(
        root_dir=data_dir,
        split='train',
        transform=get_transforms('train', img_size)
    )
    val_dataset = RareDiseaseDataset(
        root_dir=data_dir,
        split='val',
        transform=get_transforms('val', img_size)
    )

    sampler = get_weighted_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,         # weighted — replaces shuffle=True
        num_workers=4,
        pin_memory=True          # faster GPU transfer
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader, train_dataset, val_dataset