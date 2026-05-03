# src/quality_gate.py
# Quality gate: filters generated images using FID + SSIM
#
# Pipeline:
#   1. Compute SSIM between each generated image and its nearest real neighbour
#      → removes structurally broken images (blurry blobs, artifacts)
#   2. Compute FID between full generated set and real set per class
#      → measures overall distribution realism
#   3. Images passing SSIM threshold enter the gated pool
#   4. Report acceptance rate per class — this is a key result for the report

import os, sys, json, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance

# ── Config ────────────────────────────────────────────────────────────────────
REAL_DIR      = 'data/processed'
GENERATED_DIR = 'outputs/generated'
GATED_DIR     = 'outputs/gated'
REPORT_DIR    = 'outputs/quality_gate'
SEVERITIES    = ['mild', 'moderate', 'severe']
IMG_SIZE      = 128
SSIM_THRESHOLD = 0.35   # images below this are structurally too poor
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(REPORT_DIR, exist_ok=True)
for sev in SEVERITIES:
    os.makedirs(os.path.join(GATED_DIR, sev), exist_ok=True)


# ── Image loading utilities ───────────────────────────────────────────────────
def load_image_tensor(path, size=IMG_SIZE):
    """Load image as float32 tensor [3, H, W] in [0, 1]."""
    img = Image.open(path).convert('RGB').resize((size, size))
    return torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255.0


def load_folder(folder, size=IMG_SIZE):
    """Load all images in a folder as a tensor [N, 3, H, W]."""
    files = sorted([f for f in os.listdir(folder)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    tensors = [load_image_tensor(os.path.join(folder, f), size) for f in tqdm(files, desc=f'  loading {os.path.basename(folder)}', leave=False)]
    return torch.stack(tensors), files


# ── SSIM filtering ────────────────────────────────────────────────────────────
def compute_ssim_scores(real_imgs, gen_imgs):
    """
    For each generated image, compute max SSIM against all real images.
    Returns array of SSIM scores, one per generated image.

    We use max (nearest neighbour) rather than mean because we want to
    check if the generated image resembles ANY real image, not all of them.
    """
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    scores  = []

    real_imgs = real_imgs.to(DEVICE)

    for i in tqdm(range(len(gen_imgs)), desc='  computing SSIM', leave=False):
        gen = gen_imgs[i].unsqueeze(0).to(DEVICE)   # [1, 3, H, W]

        # compare against all real images, take max
        batch_scores = []
        for j in range(0, len(real_imgs), 32):
            real_batch = real_imgs[j:j+32]
            gen_batch  = gen.expand(len(real_batch), -1, -1, -1)
            score      = ssim_fn(gen_batch, real_batch).item()
            batch_scores.append(score)

        scores.append(max(batch_scores))

    return np.array(scores)


# ── FID computation ───────────────────────────────────────────────────────────
def compute_fid(real_imgs, gen_imgs):
    """
    Compute FID between real and generated image distributions.
    Lower FID = more realistic generated images.
    Medical images: FID < 100 is acceptable, < 60 is good.
    """
    fid_fn = FrechetInceptionDistance(feature=64, normalize=True).to(DEVICE)

    # add real images
    for i in range(0, len(real_imgs), 32):
        batch = real_imgs[i:i+32].to(DEVICE)
        fid_fn.update(batch, real=True)

    # add generated images
    for i in range(0, len(gen_imgs), 32):
        batch = gen_imgs[i:i+32].to(DEVICE)
        fid_fn.update(batch, real=False)

    return fid_fn.compute().item()


# ── Visualisation ─────────────────────────────────────────────────────────────
def plot_ssim_distribution(ssim_scores, threshold, severity, accepted):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(ssim_scores, bins=40, color='#4A90D9', edgecolor='white', alpha=0.8)
    ax.axvline(threshold, color='#D94A4A', linestyle='--', linewidth=2,
               label=f'Threshold = {threshold}')
    ax.set_title(f'SSIM distribution — {severity}\n'
                 f'Accepted: {accepted}/{len(ssim_scores)} '
                 f'({100*accepted/len(ssim_scores):.1f}%)', fontsize=11)
    ax.set_xlabel('SSIM score (max over real images)')
    ax.set_ylabel('Count')
    ax.legend()
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, f'ssim_dist_{severity}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_accepted_samples(gated_dir, severity, n=8):
    """Visual spot check — show accepted images side by side."""
    folder = os.path.join(gated_dir, severity)
    files  = sorted(os.listdir(folder))[:n]
    if not files:
        return

    fig, axes = plt.subplots(1, len(files), figsize=(len(files) * 2.2, 2.5))
    fig.suptitle(f'Accepted samples — {severity}', fontsize=10)
    if len(files) == 1:
        axes = [axes]
    for ax, fname in zip(axes, files):
        img = Image.open(os.path.join(folder, fname)).convert('RGB')
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, f'accepted_samples_{severity}.png')
    plt.savefig(path, dpi=120)
    plt.close()


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run_quality_gate():
    print("=" * 55)
    print("Phase 4 — Quality Gate: FID + SSIM filtering")
    print("=" * 55)
    print(f"SSIM threshold : {SSIM_THRESHOLD}")
    print(f"Device         : {DEVICE}\n")

    results = {}

    for severity in SEVERITIES:
        print(f"\n── {severity.upper()} ─────────────────────────────────────")

        real_folder = os.path.join(REAL_DIR,      severity)
        gen_folder  = os.path.join(GENERATED_DIR, severity)
        gate_folder = os.path.join(GATED_DIR,     severity)

        # load images
        print("  Loading real images...")
        real_imgs, real_files = load_folder(real_folder)
        print(f"  Real images    : {len(real_imgs)}")

        print("  Loading generated images...")
        gen_imgs, gen_files = load_folder(gen_folder)
        print(f"  Generated      : {len(gen_imgs)}")

        # ── SSIM filter ───────────────────────────────────────────────────────
        print("  Computing SSIM scores (nearest-neighbour)...")
        ssim_scores = compute_ssim_scores(real_imgs, gen_imgs)

        mask     = ssim_scores >= SSIM_THRESHOLD
        accepted = mask.sum()
        rejected = len(mask) - accepted

        print(f"  SSIM — accepted: {accepted} / {len(gen_imgs)} "
              f"({100*accepted/len(gen_imgs):.1f}%)")
        print(f"  SSIM — rejected: {rejected}")
        print(f"  SSIM — mean    : {ssim_scores.mean():.4f}  "
              f"min: {ssim_scores.min():.4f}  max: {ssim_scores.max():.4f}")

        # copy accepted images to gated folder
        for i, (fname, keep) in enumerate(zip(gen_files, mask)):
            if keep:
                shutil.copy(
                    os.path.join(gen_folder, fname),
                    os.path.join(gate_folder, fname)
                )

        # ── FID — full generated set vs real ─────────────────────────────────
        print("  Computing FID (full generated set vs real)...")
        fid_all = compute_fid(real_imgs, gen_imgs)

        # FID — gated set vs real
        if accepted > 0:
            gated_imgs = gen_imgs[mask]
            print("  Computing FID (gated set vs real)...")
            fid_gated = compute_fid(real_imgs, gated_imgs)
        else:
            fid_gated = float('inf')

        print(f"  FID (all generated) : {fid_all:.2f}")
        print(f"  FID (gated only)    : {fid_gated:.2f}")

        # ── plots ─────────────────────────────────────────────────────────────
        plot_ssim_distribution(ssim_scores, SSIM_THRESHOLD, severity, accepted)
        plot_accepted_samples(GATED_DIR, severity)

        results[severity] = {
            'real_count'     : int(len(real_imgs)),
            'generated_count': int(len(gen_imgs)),
            'accepted'       : int(accepted),
            'rejected'       : int(rejected),
            'acceptance_rate': float(accepted / len(gen_imgs)),
            'ssim_mean'      : float(ssim_scores.mean()),
            'ssim_min'       : float(ssim_scores.min()),
            'ssim_max'       : float(ssim_scores.max()),
            'fid_all'        : float(fid_all),
            'fid_gated'      : float(fid_gated),
        }

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("QUALITY GATE SUMMARY")
    print("=" * 55)
    print(f"{'Severity':<12} {'Real':>6} {'Gen':>6} {'Accept':>8} "
          f"{'Rate':>7} {'SSIM':>7} {'FID-all':>9} {'FID-gate':>9}")
    print("-" * 65)

    total_accepted = 0
    for sev, r in results.items():
        print(f"{sev:<12} {r['real_count']:>6} {r['generated_count']:>6} "
              f"{r['accepted']:>8} {r['acceptance_rate']:>6.1%} "
              f"{r['ssim_mean']:>7.4f} {r['fid_all']:>9.2f} "
              f"{r['fid_gated']:>9.2f}")
        total_accepted += r['accepted']

    print("-" * 65)
    print(f"{'TOTAL':<12} {'':>6} {'1500':>6} {total_accepted:>8}")

    # ── Save results JSON ─────────────────────────────────────────────────────
    with open(os.path.join(REPORT_DIR, 'quality_gate_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved → {REPORT_DIR}/quality_gate_results.json")
    print(f"Gated images  → {GATED_DIR}/")
    print(f"Plots         → {REPORT_DIR}/")
    print("\nPhase 4 complete. Ready for Phase 5 — classifier training.")

    return results


if __name__ == '__main__':
    run_quality_gate()