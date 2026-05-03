# src/classifier.py
# Complete Phase 5 — Classifier Training and Evaluation
#
# Three experiments:
#   1. Baseline        — real images only (492)
#   2. CVAE-augmented  — real + CVAE decoder samples
#   3. cGAN-augmented  — real + quality-gated cGAN images (1958)
#
# Metrics per experiment:
#   - Accuracy, Precision, Recall, F1 (macro + per class)
#   - Confusion matrix
#   - ROC curves (one-vs-rest per severity class)
#   - Training curves
#
# Final outputs:
#   - Side-by-side comparison bar chart (key report figure)
#   - Per-class F1 breakdown
#   - Full results JSON

import os, sys, time, json, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
from collections import Counter
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = {
    'real_dir'      : 'data/processed',
    'pool_dir'      : 'outputs/augmented_pool',
    'cvae_dir'      : 'outputs/cvae_samples',
    'output_dir'    : 'outputs/classifier',
    'model_dir'     : 'models',
    'img_size'      : 128,
    'batch_size'    : 32,
    'epochs'        : 50,
    'lr'            : 1e-4,
    'weight_decay'  : 1e-4,
    'val_split'     : 0.20,
    'num_classes'   : 3,
    'seeds'         : [42],   # add [42,7,21] for multi-seed if time allows
}

SEVERITY_NAMES = ['mild', 'moderate', 'severe']
SEVERITY_MAP   = {'mild': 0, 'moderate': 1, 'severe': 2}
COLORS         = {'Baseline': '#4A90D9',
                  'CVAE-Aug': '#E8A838',
                  'cGAN-Aug': '#4AD94A'}

os.makedirs(CONFIG['output_dir'], exist_ok=True)
os.makedirs(CONFIG['model_dir'],  exist_ok=True)


# ── Step 0: Generate CVAE samples for experiment 2 ───────────────────────────
def generate_cvae_samples(cfg, device):
    """
    Samples images from the trained CVAE decoder for each severity class.
    These are used in experiment 2 (CVAE-augmented) to show CVAE-only
    augmentation vs cGAN augmentation.
    """
    from src.model   import CVAE
    from src.dataset import get_dataloaders

    cvae_dir = cfg['cvae_dir']
    for sev in SEVERITY_NAMES:
        os.makedirs(os.path.join(cvae_dir, sev), exist_ok=True)

    # check if already generated
    total = sum(
        len(os.listdir(os.path.join(cvae_dir, s))) for s in SEVERITY_NAMES
    )
    if total >= 900:
        print(f"  CVAE samples already exist ({total} images) — skipping generation")
        return

    print("  Loading trained CVAE...")
    checkpoint = torch.load('models/cvae_best.pth', map_location=device)
    model_cfg  = checkpoint['config']
    cvae       = CVAE(model_cfg['latent_dim'], model_cfg['num_classes']).to(device)
    cvae.load_state_dict(checkpoint['model_state_dict'])
    cvae.eval()

    # extract per-severity cluster stats from training data
    train_loader, _, _, _ = get_dataloaders(
        cfg['real_dir'], cfg['img_size'], cfg['batch_size']
    )

    all_mu = {0: [], 1: [], 2: []}
    with torch.no_grad():
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            mu, _        = cvae.encoder(imgs, labels)
            for s in range(3):
                mask = labels == s
                if mask.sum():
                    all_mu[s].append(mu[mask].cpu().numpy())

    cluster_mu  = {s: np.concatenate(all_mu[s]).mean(axis=0) for s in range(3)}
    cluster_std = {s: np.concatenate(all_mu[s]).std(axis=0)  for s in range(3)}

    rng      = np.random.RandomState(42)
    n_per_class = 300   # 300 CVAE samples per class

    print(f"  Generating {n_per_class} CVAE samples per severity class...")
    with torch.no_grad():
        for sev_idx, sev_name in enumerate(SEVERITY_NAMES):
            mu  = cluster_mu[sev_idx]
            std = cluster_std[sev_idx]
            for i in range(n_per_class):
                eps  = rng.randn(model_cfg['latent_dim']).astype(np.float32)
                z    = torch.tensor(mu + eps * std).unsqueeze(0).to(device)
                lbl  = torch.tensor([sev_idx], device=device)
                img  = cvae.decode(z, lbl)
                img  = (img.clamp(-1, 1) + 1) / 2 * 255
                img  = img[0].permute(1,2,0).cpu().numpy().astype(np.uint8)
                Image.fromarray(img).save(
                    os.path.join(cvae_dir, sev_name, f'cvae_{i:04d}.png')
                )
    print(f"  CVAE samples saved to {cvae_dir}/")


# ── Dataset ───────────────────────────────────────────────────────────────────
class SkinLesionDataset(Dataset):
    """
    Loads images from a folder structure: root/{mild,moderate,severe}/*.png
    Supports train/val split. Prefix filtering allows mixing real + synthetic.
    """
    def __init__(self, root_dir, split='train', val_split=0.20,
                 transform=None, seed=42, extra_dir=None):
        """
        root_dir  : primary data directory
        extra_dir : optional second directory to merge into training set
                    (used to add CVAE samples on top of real images)
        """
        self.transform = transform
        all_samples    = []

        for directory in ([root_dir] + ([extra_dir] if extra_dir else [])):
            for severity, idx in SEVERITY_MAP.items():
                folder = os.path.join(directory, severity)
                if not os.path.exists(folder):
                    continue
                for fname in sorted(os.listdir(folder)):
                    if fname.lower().endswith(('.jpg','.jpeg','.png')):
                        all_samples.append(
                            (os.path.join(folder, fname), idx)
                        )

        rng     = np.random.RandomState(seed)
        indices = rng.permutation(len(all_samples))
        val_n   = int(len(all_samples) * val_split)

        if split == 'val':
            selected = indices[:val_n]
        else:
            selected = indices[val_n:]

        self.samples = [all_samples[i] for i in selected]
        counts       = Counter(s[1] for s in self.samples)
        print(f"    [{split:5s}] {len(self.samples):4d} images — "
              f"mild: {counts[0]:3d}, "
              f"moderate: {counts[1]:3d}, "
              f"severe: {counts[2]:3d}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


def get_transforms(split, img_size):
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.1, 0.1, 0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])


def get_weighted_sampler(dataset):
    labels        = [s[1] for s in dataset.samples]
    counts        = Counter(labels)
    total         = sum(counts.values())
    class_weights = {c: total / cnt for c, cnt in counts.items()}
    weights       = [class_weights[l] for l in labels]
    return WeightedRandomSampler(weights, len(weights), replacement=True)


# ── Model ─────────────────────────────────────────────────────────────────────
def get_model(num_classes, device):
    model    = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


# ── Training ──────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct    += (out.argmax(1) == labels).sum().item()
        total      += labels.shape[0]
    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for imgs, labels in loader:
        imgs     = imgs.to(device)
        logits   = model(imgs)
        probs    = torch.softmax(logits, dim=1).cpu().numpy()
        preds    = logits.argmax(1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
        all_probs.append(probs)

    all_probs = np.concatenate(all_probs, axis=0)
    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro',
                           zero_division=0)
    rec  = recall_score(all_labels, all_preds, average='macro',
                        zero_division=0)
    f1   = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_per_class = f1_score(all_labels, all_preds, average=None,
                            zero_division=0)
    cm   = confusion_matrix(all_labels, all_preds, labels=[0,1,2])

    return (acc, prec, rec, f1, f1_per_class, cm,
            all_preds, all_labels, all_probs)


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_confusion_matrix(cm, title, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=SEVERITY_NAMES,
                yticklabels=SEVERITY_NAMES, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_roc_curves(all_labels, all_probs, experiment_name, path):
    """One-vs-rest ROC curve per severity class."""
    y_bin = label_binarize(all_labels, classes=[0, 1, 2])
    colors = ['#4A90D9', '#E8A838', '#D94A4A']

    fig, ax = plt.subplots(figsize=(7, 6))
    for i, (sev, col) in enumerate(zip(SEVERITY_NAMES, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=col, lw=2,
                label=f'{sev} (AUC = {roc_auc:.3f})')

    ax.plot([0,1], [0,1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC curves (one-vs-rest) — {experiment_name}', fontsize=11)
    ax.legend(loc='lower right')
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_training_curves(train_losses, val_accs, title, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(train_losses, color='#4A90D9')
    axes[0].set_title('Training loss')
    axes[0].set_xlabel('Epoch')
    axes[1].plot(val_accs, color='#4AD94A')
    axes[1].set_title('Validation accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylim(0, 1)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
    plt.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_final_comparison(all_results):
    """Three-way comparison: Baseline vs CVAE-Aug vs cGAN-Aug."""

    # ── macro metrics bar chart ────────────────────────────────────────────
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    x       = np.arange(len(metrics))
    width   = 0.25
    offsets = [-width, 0, width]

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (name, res) in enumerate(all_results.items()):
        vals = [res['accuracy'], res['precision'],
                res['recall'],   res['f1']]
        bars = ax.bar(x + offsets[i], vals, width,
                      label=name, color=COLORS[name], alpha=0.85)
        ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score')
    ax.set_title('Classifier comparison: Baseline vs CVAE-Aug vs cGAN-Aug',
                 fontsize=12)
    ax.legend()
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{CONFIG['output_dir']}/final_comparison_macro.png", dpi=150)
    plt.close()

    # ── per-class F1 chart ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    x2      = np.arange(len(SEVERITY_NAMES))

    for i, (name, res) in enumerate(all_results.items()):
        bars = ax.bar(x2 + offsets[i], res['f1_per_class'], width,
                      label=name, color=COLORS[name], alpha=0.85)
        ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=8)

    ax.set_xticks(x2)
    ax.set_xticklabels(SEVERITY_NAMES)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('F1 Score')
    ax.set_title('Per-class F1: Baseline vs CVAE-Aug vs cGAN-Aug', fontsize=12)
    ax.legend()
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{CONFIG['output_dir']}/final_comparison_per_class_f1.png",
                dpi=150)
    plt.close()


# ── Single experiment runner ──────────────────────────────────────────────────
def run_experiment(name, train_dir, cfg, device,
                   extra_train_dir=None, seed=42):
    """
    Trains ResNet-18 and evaluates on real-only val set.

    train_dir       : primary training data directory
    extra_train_dir : optional extra training data (CVAE samples)
    The val set is ALWAYS real images only for fair comparison.
    """
    print(f"\n{'─'*55}")
    print(f"Experiment : {name}  (seed={seed})")
    print(f"{'─'*55}")

    train_ds = SkinLesionDataset(
        train_dir, split='train', val_split=cfg['val_split'],
        transform=get_transforms('train', cfg['img_size']),
        seed=seed, extra_dir=extra_train_dir
    )
    # val is ALWAYS real-only, same split every time
    val_ds = SkinLesionDataset(
        cfg['real_dir'], split='val', val_split=cfg['val_split'],
        transform=get_transforms('val', cfg['img_size']),
        seed=seed
    )

    sampler      = get_weighted_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'],
                              sampler=sampler, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg['batch_size'],
                              shuffle=False, num_workers=4, pin_memory=True)

    model     = get_model(cfg['num_classes'], device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['epochs']
    )

    train_losses, val_accs = [], []
    best_val_acc           = 0.0
    best_path = f"{cfg['model_dir']}/cls_{name.lower().replace('-','_')}_s{seed}.pth"
    log_path  = f"{cfg['output_dir']}/{name.lower().replace('-','_')}_s{seed}_log.csv"

    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch','train_loss','val_acc','val_f1'])

    print(f"\n{'Epoch':>6} {'Loss':>8} {'ValAcc':>8} {'ValF1':>8}")
    print("-" * 36)

    for epoch in range(1, cfg['epochs'] + 1):
        t0      = time.time()
        loss, _ = train_one_epoch(model, train_loader,
                                  criterion, optimizer, device)
        (acc, _, _, f1, _, _, _, _, _) = evaluate(model, val_loader, device)
        scheduler.step()

        train_losses.append(loss)
        val_accs.append(acc)

        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, f'{loss:.4f}',
                                    f'{acc:.4f}', f'{f1:.4f}'])

        elapsed = time.time() - t0
        print(f"{epoch:>6} {loss:>8.4f} {acc:>8.4f} {f1:>8.4f}  ({elapsed:.1f}s)")

        if acc > best_val_acc:
            best_val_acc = acc
            torch.save(model.state_dict(), best_path)

    # final evaluation
    model.load_state_dict(torch.load(best_path, map_location=device))
    (acc, prec, rec, f1,
     f1_pc, cm, preds,
     labels, probs) = evaluate(model, val_loader, device)

    print(f"\n── {name} Final Results ──")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}  (macro)")
    print(f"  Recall    : {rec:.4f}  (macro)")
    print(f"  F1        : {f1:.4f}  (macro)")
    print(f"  F1 mild   : {f1_pc[0]:.4f}")
    print(f"  F1 mod    : {f1_pc[1]:.4f}")
    print(f"  F1 severe : {f1_pc[2]:.4f}")
    print(classification_report(labels, preds,
                                target_names=SEVERITY_NAMES,
                                zero_division=0))

    # plots
    safe_name = name.lower().replace('-', '_')
    plot_confusion_matrix(
        cm, f'Confusion matrix — {name}',
        f"{cfg['output_dir']}/cm_{safe_name}.png"
    )
    plot_roc_curves(
        labels, probs, name,
        f"{cfg['output_dir']}/roc_{safe_name}.png"
    )
    plot_training_curves(
        train_losses, val_accs, f'Training curves — {name}',
        f"{cfg['output_dir']}/curves_{safe_name}.png"
    )

    return {
        'name'        : name,
        'accuracy'    : float(acc),
        'precision'   : float(prec),
        'recall'      : float(rec),
        'f1'          : float(f1),
        'f1_per_class': [float(x) for x in f1_pc],
        'best_val_acc': float(best_val_acc),
        'cm'          : cm.tolist(),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    cfg    = CONFIG
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")
    if device.type == 'cuda':
        print(f"GPU    : {torch.cuda.get_device_name(0)}\n")

    # Step 0: generate CVAE samples for experiment 2
    print("Step 0: Preparing CVAE samples for experiment 2...")
    generate_cvae_samples(cfg, device)

    all_results = {}

    # Experiment 1: Baseline — real images only
    all_results['Baseline'] = run_experiment(
        name      = 'Baseline',
        train_dir = cfg['real_dir'],
        cfg       = cfg,
        device    = device,
    )

    # Experiment 2: CVAE-augmented — real + CVAE decoder samples
    all_results['CVAE-Aug'] = run_experiment(
        name           = 'CVAE-Aug',
        train_dir      = cfg['real_dir'],
        extra_train_dir= cfg['cvae_dir'],
        cfg            = cfg,
        device         = device,
    )

    # Experiment 3: cGAN-augmented — real + quality-gated cGAN images
    all_results['cGAN-Aug'] = run_experiment(
        name      = 'cGAN-Aug',
        train_dir = cfg['pool_dir'],
        cfg       = cfg,
        device    = device,
    )

    # final comparison plots
    plot_final_comparison(all_results)

    # save all results
    save_results = {k: {m: v for m, v in r.items() if m != 'cm'}
                    for k, r in all_results.items()}
    with open(f"{cfg['output_dir']}/classifier_results.json", 'w') as f:
        json.dump(save_results, f, indent=2)

    # ── Final comparison table ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("FINAL COMPARISON — ALL THREE EXPERIMENTS")
    print("=" * 65)
    print(f"{'Metric':<14} {'Baseline':>10} {'CVAE-Aug':>10} {'cGAN-Aug':>10} {'Best':>8}")
    print("-" * 58)

    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        vals     = {k: all_results[k][metric] for k in all_results}
        best_key = max(vals, key=vals.get)
        print(f"{metric:<14} "
              f"{vals['Baseline']:>10.4f} "
              f"{vals['CVAE-Aug']:>10.4f} "
              f"{vals['cGAN-Aug']:>10.4f}  "
              f"{best_key:>8}")

    print("\nPer-class F1:")
    print(f"{'Class':<14} {'Baseline':>10} {'CVAE-Aug':>10} {'cGAN-Aug':>10}")
    print("-" * 48)
    for i, sev in enumerate(SEVERITY_NAMES):
        vals = {k: all_results[k]['f1_per_class'][i] for k in all_results}
        print(f"{sev:<14} "
              f"{vals['Baseline']:>10.4f} "
              f"{vals['CVAE-Aug']:>10.4f} "
              f"{vals['cGAN-Aug']:>10.4f}")

    print("\n" + "=" * 65)
    print("Outputs saved to outputs/classifier/:")
    print("  cm_baseline.png            — confusion matrix")
    print("  cm_cvae_aug.png            — confusion matrix")
    print("  cm_cgan_aug.png            — confusion matrix")
    print("  roc_baseline.png           — ROC curves")
    print("  roc_cvae_aug.png           — ROC curves")
    print("  roc_cgan_aug.png           — ROC curves")
    print("  curves_baseline.png        — training curves")
    print("  curves_cvae_aug.png        — training curves")
    print("  curves_cgan_aug.png        — training curves")
    print("  final_comparison_macro.png ← KEY REPORT FIGURE")
    print("  final_comparison_per_class_f1.png ← KEY REPORT FIGURE")
    print("  classifier_results.json    — all metrics")
    print("=" * 65)


if __name__ == '__main__':
    main()
