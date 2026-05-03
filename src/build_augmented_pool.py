# src/build_augmented_pool.py
# Assembles the final augmented training pool: real + gated synthetic images
# This is what the Phase 5 classifier will train on.
#
# Output structure:
#   outputs/augmented_pool/
#       mild/        ← real mild + gated synthetic mild
#       moderate/    ← real moderate + gated synthetic moderate  
#       severe/      ← real severe + gated synthetic severe
#       pool_stats.json  ← counts for report

import os, sys, shutil, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

REAL_DIR    = 'data/processed'
GATED_DIR   = 'outputs/gated'
POOL_DIR    = 'outputs/augmented_pool'
REPORT_DIR  = 'outputs/quality_gate'
SEVERITIES  = ['mild', 'moderate', 'severe']
COLORS      = {'mild': '#4A90D9', 'moderate': '#E8A838', 'severe': '#D94A4A'}

os.makedirs(REPORT_DIR, exist_ok=True)
for sev in SEVERITIES:
    os.makedirs(os.path.join(POOL_DIR, sev), exist_ok=True)


def build_pool():
    print("=" * 55)
    print("Building augmented training pool")
    print("=" * 55)

    stats = {}
    total_real = 0
    total_synthetic = 0

    for severity in SEVERITIES:
        real_folder  = os.path.join(REAL_DIR,  severity)
        gated_folder = os.path.join(GATED_DIR, severity)
        pool_folder  = os.path.join(POOL_DIR,  severity)

        # count real images
        real_files = [f for f in os.listdir(real_folder)
                      if f.lower().endswith(('.jpg','.jpeg','.png'))]

        # count gated synthetic images
        gated_files = [f for f in os.listdir(gated_folder)
                       if f.lower().endswith(('.jpg','.jpeg','.png'))]

        # copy real images
        print(f"\n{severity}: copying {len(real_files)} real images...")
        for fname in tqdm(real_files, leave=False):
            shutil.copy(
                os.path.join(real_folder, fname),
                os.path.join(pool_folder, f'real_{fname}')
            )

        # copy gated synthetic images
        print(f"{severity}: copying {len(gated_files)} gated synthetic images...")
        for fname in tqdm(gated_files, leave=False):
            shutil.copy(
                os.path.join(gated_folder, fname),
                os.path.join(pool_folder, f'synth_{fname}')
            )

        total_in_pool = len(real_files) + len(gated_files)
        stats[severity] = {
            'real'       : len(real_files),
            'synthetic'  : len(gated_files),
            'total'      : total_in_pool,
            'augmentation_ratio': round(len(gated_files) / max(len(real_files), 1), 2)
        }

        total_real      += len(real_files)
        total_synthetic += len(gated_files)

        print(f"  {severity}: {len(real_files)} real + "
              f"{len(gated_files)} synthetic = {total_in_pool} total")

    stats['_totals'] = {
        'real'     : total_real,
        'synthetic': total_synthetic,
        'total'    : total_real + total_synthetic
    }

    # save stats
    with open(os.path.join(POOL_DIR, 'pool_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    # ── Before / After augmentation bar chart ─────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Dataset augmentation: before vs after', fontsize=12)

    # before
    before = [stats[s]['real'] for s in SEVERITIES]
    axes[0].bar(SEVERITIES, before,
                color=[COLORS[s] for s in SEVERITIES], width=0.5)
    axes[0].bar_label(axes[0].containers[0], padding=4)
    axes[0].set_title('Before augmentation (real only)')
    axes[0].set_ylabel('Number of images')
    axes[0].set_ylim(0, max(stats[s]['total'] for s in SEVERITIES) * 1.15)
    axes[0].spines[['top','right']].set_visible(False)

    # after — stacked bar
    real_counts  = [stats[s]['real']      for s in SEVERITIES]
    synth_counts = [stats[s]['synthetic'] for s in SEVERITIES]
    axes[1].bar(SEVERITIES, real_counts,
                color=[COLORS[s] for s in SEVERITIES],
                width=0.5, label='Real', alpha=0.9)
    axes[1].bar(SEVERITIES, synth_counts,
                bottom=real_counts,
                color=[COLORS[s] for s in SEVERITIES],
                width=0.5, label='Synthetic', alpha=0.4,
                hatch='//')
    axes[1].set_title('After augmentation (real + synthetic)')
    axes[1].set_ylabel('Number of images')
    axes[1].set_ylim(0, max(stats[s]['total'] for s in SEVERITIES) * 1.15)
    axes[1].legend()
    axes[1].spines[['top','right']].set_visible(False)

    plt.tight_layout()
    path = os.path.join(REPORT_DIR, 'augmentation_comparison.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nAugmentation chart saved → {path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("AUGMENTED POOL SUMMARY")
    print("=" * 55)
    print(f"{'Severity':<12} {'Real':>6} {'Synth':>7} {'Total':>7} {'Ratio':>7}")
    print("-" * 42)
    for sev in SEVERITIES:
        r = stats[sev]
        print(f"{sev:<12} {r['real']:>6} {r['synthetic']:>7} "
              f"{r['total']:>7} {r['augmentation_ratio']:>6.1f}x")
    print("-" * 42)
    print(f"{'TOTAL':<12} {total_real:>6} {total_synthetic:>7} "
          f"{total_real+total_synthetic:>7}")
    print(f"\nPool saved → {POOL_DIR}/")
    print("Pool stats → {POOL_DIR}/pool_stats.json")
    return stats


if __name__ == '__main__':
    build_pool()