# src/eda.py
# Full exploratory data analysis — generates all figures for the report
# Run once after Phase 1 labelling is complete

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from collections import defaultdict
import cv2
from tqdm import tqdm

DATA_DIR   = 'data/processed'
OUTPUT_DIR = 'outputs/eda'
SEVERITIES = ['mild', 'moderate', 'severe']
COLORS     = {'mild': '#4A90D9', 'moderate': '#E8A838', 'severe': '#D94A4A'}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. Class distribution bar chart ─────────────────────────────────────────
def plot_class_distribution():
    counts = {}
    for sev in SEVERITIES:
        folder = os.path.join(DATA_DIR, sev)
        counts[sev] = len([f for f in os.listdir(folder)
                           if f.lower().endswith(('.jpg','.jpeg','.png'))])

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.keys(), counts.values(),
                  color=[COLORS[s] for s in counts.keys()], width=0.5)
    ax.bar_label(bars, padding=4, fontsize=11)
    ax.set_title('Image count per severity class', fontsize=13)
    ax.set_ylabel('Number of images')
    ax.set_ylim(0, max(counts.values()) * 1.2)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/class_distribution.png', dpi=150)
    plt.close()
    print(f"Saved class distribution → {OUTPUT_DIR}/class_distribution.png")
    return counts


# ── 2. Sample image grid ─────────────────────────────────────────────────────
def plot_sample_grid(n_per_class=6):
    fig, axes = plt.subplots(3, n_per_class, figsize=(n_per_class * 2.5, 8))
    fig.suptitle('Sample images per severity class', fontsize=13, y=1.01)

    for row, sev in enumerate(SEVERITIES):
        folder = os.path.join(DATA_DIR, sev)
        files  = sorted(os.listdir(folder))[:n_per_class]
        for col, fname in enumerate(files):
            img = Image.open(os.path.join(folder, fname)).convert('RGB')
            img = img.resize((128, 128))
            axes[row][col].imshow(img)
            axes[row][col].axis('off')
            if col == 0:
                axes[row][col].set_ylabel(sev, fontsize=11,
                                          rotation=0, labelpad=50,
                                          va='center')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/sample_grid.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved sample grid → {OUTPUT_DIR}/sample_grid.png")


# ── 3. Image size distribution ───────────────────────────────────────────────
def plot_size_distribution():
    widths, heights = [], []
    for sev in SEVERITIES:
        folder = os.path.join(DATA_DIR, sev)
        for fname in os.listdir(folder):
            if not fname.lower().endswith(('.jpg','.jpeg','.png')):
                continue
            img = Image.open(os.path.join(folder, fname))
            w, h = img.size
            widths.append(w)
            heights.append(h)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(widths,  bins=20, color='#4A90D9', edgecolor='white')
    axes[0].set_title('Image widths (px)')
    axes[0].set_xlabel('Width')
    axes[1].hist(heights, bins=20, color='#E8A838', edgecolor='white')
    axes[1].set_title('Image heights (px)')
    axes[1].set_xlabel('Height')
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
    plt.suptitle('Raw image size distribution before resizing', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/size_distribution.png', dpi=150)
    plt.close()
    print(f"Saved size distribution → {OUTPUT_DIR}/size_distribution.png")


# ── 4. Colour statistics per severity class ──────────────────────────────────
def plot_colour_stats():
    """
    Plots mean RGB channel values per severity class.
    Shows whether severity classes are visually distinguishable by colour.
    This validates our colour-variance based severity proxy from Phase 1.
    """
    stats = defaultdict(lambda: {'r': [], 'g': [], 'b': []})

    for sev in SEVERITIES:
        folder = os.path.join(DATA_DIR, sev)
        files  = [f for f in os.listdir(folder)
                  if f.lower().endswith(('.jpg','.jpeg','.png'))]
        print(f"Computing colour stats for {sev} ({len(files)} images)...")
        for fname in tqdm(files):
            img = np.array(
                Image.open(os.path.join(folder, fname))
                .convert('RGB').resize((128, 128))
            ).astype(float)
            stats[sev]['r'].append(img[:,:,0].mean())
            stats[sev]['g'].append(img[:,:,1].mean())
            stats[sev]['b'].append(img[:,:,2].mean())

    channels = ['r', 'g', 'b']
    ch_colors = ['#D94A4A', '#4AD94A', '#4A4AD9']
    x = np.arange(len(SEVERITIES))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (ch, col) in enumerate(zip(channels, ch_colors)):
        means = [np.mean(stats[sev][ch]) for sev in SEVERITIES]
        stds  = [np.std(stats[sev][ch])  for sev in SEVERITIES]
        ax.bar(x + i*width, means, width, yerr=stds,
               label=f'{ch.upper()} channel', color=col, alpha=0.8,
               capsize=4, error_kw={'linewidth': 1})

    ax.set_xticks(x + width)
    ax.set_xticklabels(SEVERITIES)
    ax.set_ylabel('Mean pixel value')
    ax.set_title('Mean RGB channel values per severity class', fontsize=12)
    ax.legend()
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/colour_stats.png', dpi=150)
    plt.close()
    print(f"Saved colour stats → {OUTPUT_DIR}/colour_stats.png")


# ── 5. Colour variance distribution (validates severity proxy) ───────────────
def plot_variance_distribution():
    """
    Plots the HSV saturation variance distribution per class.
    This is the exact metric used to assign severity labels in Phase 1.
    A good label assignment shows increasing variance: mild < moderate < severe.
    """
    variances = {sev: [] for sev in SEVERITIES}

    for sev in SEVERITIES:
        folder = os.path.join(DATA_DIR, sev)
        files  = [f for f in os.listdir(folder)
                  if f.lower().endswith(('.jpg','.jpeg','.png'))]
        for fname in files:
            img = cv2.imread(os.path.join(folder, fname))
            if img is None:
                continue
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            variances[sev].append(float(np.var(hsv[:,:,1])))

    fig, ax = plt.subplots(figsize=(8, 5))
    for sev in SEVERITIES:
        ax.hist(variances[sev], bins=25, alpha=0.6,
                label=sev, color=COLORS[sev], edgecolor='white')
    ax.set_xlabel('HSV saturation variance')
    ax.set_ylabel('Number of images')
    ax.set_title('Saturation variance per severity class\n'
                 '(validates severity proxy used in labelling)', fontsize=11)
    ax.legend()
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/variance_distribution.png', dpi=150)
    plt.close()
    print(f"Saved variance distribution → {OUTPUT_DIR}/variance_distribution.png")


# ── 6. Pixel intensity histograms ────────────────────────────────────────────
def plot_pixel_histograms():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    fig.suptitle('Pixel intensity distribution per severity class', fontsize=12)

    for ax, sev in zip(axes, SEVERITIES):
        folder = os.path.join(DATA_DIR, sev)
        files  = [f for f in os.listdir(folder)
                  if f.lower().endswith(('.jpg','.jpeg','.png'))]
        all_pixels = []
        for fname in files[:50]:   # sample 50 per class for speed
            img = np.array(
                Image.open(os.path.join(folder, fname))
                .convert('L').resize((128, 128))
            ).flatten()
            all_pixels.extend(img.tolist())

        ax.hist(all_pixels, bins=50, color=COLORS[sev],
                edgecolor='none', alpha=0.85)
        ax.set_title(sev)
        ax.set_xlabel('Pixel intensity')
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/pixel_histograms.png', dpi=150)
    plt.close()
    print(f"Saved pixel histograms → {OUTPUT_DIR}/pixel_histograms.png")


# ── Run all ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 50)
    print("Running full EDA pipeline")
    print("=" * 50)

    counts = plot_class_distribution()
    print(f"\nDataset summary:")
    for sev, cnt in counts.items():
        print(f"  {sev:>10} : {cnt} images")
    print(f"  {'TOTAL':>10} : {sum(counts.values())} images")

    plot_sample_grid()
    plot_size_distribution()
    plot_colour_stats()
    plot_variance_distribution()
    plot_pixel_histograms()

    print("\n" + "=" * 50)
    print("EDA complete. All figures saved to outputs/eda/")
    print("Use these directly in your report — Section 3: Dataset Description")
    print("=" * 50)