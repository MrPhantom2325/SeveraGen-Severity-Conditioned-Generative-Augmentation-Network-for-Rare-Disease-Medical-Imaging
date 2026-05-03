# src/train.py
# Full CVAE training loop
#
# Features:
#   - Beta annealing (KL weight warms up gradually to avoid posterior collapse)
#   - Checkpoint saving (best model + every 10 epochs)
#   - Training + validation loss curves saved to outputs/
#   - Reconstruction preview grid saved every 10 epochs
#   - Early stopping if val loss doesn't improve for 15 epochs
#   - Full logging to outputs/training_log.csv

import os, sys, time, csv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.model   import CVAE, count_parameters
from src.loss    import cvae_loss
from src.dataset import get_dataloaders, IDX_TO_SEVERITY


# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = {
    'data_dir'      : 'data/processed',
    'output_dir'    : 'outputs',
    'model_dir'     : 'models',
    'img_size'      : 128,
    'batch_size'    : 32,
    'latent_dim'    : 128,
    'num_classes'   : 3,
    'epochs'        : 100,
    'lr'            : 1e-3,
    'beta_start'    : 0.0,      # KL weight at epoch 0
    'beta_end'      : 0.001,    # KL weight at epoch beta_warmup
    'beta_warmup'   : 30,       # epochs to ramp up beta
    'save_every'    : 10,       # save checkpoint every N epochs
    'early_stop'    : 15,       # stop if val loss plateaus for N epochs
    'preview_every' : 10,       # save reconstruction grid every N epochs
}


# ── Beta annealing schedule ───────────────────────────────────────────────────
def get_beta(epoch, cfg):
    """
    Linearly warm up KL weight from beta_start to beta_end
    over the first beta_warmup epochs, then hold constant.

    Why annealing?
    At the start of training, the decoder is random and can't reconstruct
    well. If KL is too strong early, the encoder collapses to N(0,I)
    immediately and never learns severity-specific clusters.
    Warming up gradually lets reconstruction stabilise first.
    """
    if epoch >= cfg['beta_warmup']:
        return cfg['beta_end']
    progress = epoch / cfg['beta_warmup']
    return cfg['beta_start'] + progress * (cfg['beta_end'] - cfg['beta_start'])


# ── Reconstruction preview ────────────────────────────────────────────────────
def save_reconstruction_preview(model, val_loader, epoch, cfg, device):
    """
    Saves a side-by-side grid: original vs reconstructed images.
    3 rows (one per severity) x 4 columns (original | reconstructed).
    Saved to outputs/previews/epoch_{N:03d}.png
    """
    preview_dir = os.path.join(cfg['output_dir'], 'previews')
    os.makedirs(preview_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        imgs, labels = next(iter(val_loader))
        imgs, labels = imgs.to(device), labels.to(device)

        x_recon, _, _ = model(imgs, labels)

    # pick one image per severity class
    imgs_np   = imgs.cpu().numpy()
    recon_np  = x_recon.cpu().numpy()
    labels_np = labels.cpu().numpy()

    fig, axes = plt.subplots(3, 8, figsize=(18, 7))
    fig.suptitle(f'Epoch {epoch} — Original (left) vs Reconstructed (right)',
                 fontsize=11)

    shown = {0: 0, 1: 0, 2: 0}
    for i in range(len(labels_np)):
        sev = int(labels_np[i])
        col_offset = shown[sev] * 2
        if shown[sev] >= 4:
            continue

        for ax_col, arr in zip([col_offset, col_offset+1],
                               [imgs_np[i], recon_np[i]]):
            img = arr.transpose(1, 2, 0)    # CHW → HWC
            img = (img * 0.5 + 0.5).clip(0, 1)
            axes[sev][ax_col].imshow(img)
            axes[sev][ax_col].axis('off')

        if shown[sev] == 0:
            axes[sev][0].set_ylabel(IDX_TO_SEVERITY[sev], fontsize=9)

        shown[sev] += 1
        if all(v >= 4 for v in shown.values()):
            break

    plt.tight_layout()
    path = os.path.join(preview_dir, f'epoch_{epoch:03d}.png')
    plt.savefig(path, dpi=100)
    plt.close()
    return path


# ── Loss curve plot ───────────────────────────────────────────────────────────
def save_loss_curves(train_losses, val_losses, recon_losses,
                     kl_losses, betas, cfg):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(train_losses, label='train total', color='#4A90D9')
    axes[0].plot(val_losses,   label='val total',   color='#D94A4A')
    axes[0].set_title('Total loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()

    axes[1].plot(recon_losses, color='#4AD94A')
    axes[1].set_title('Reconstruction loss (train)')
    axes[1].set_xlabel('Epoch')

    axes[2].plot(kl_losses, color='#E8A838', label='KL loss')
    ax2 = axes[2].twinx()
    ax2.plot(betas, color='#9B59B6', linestyle='--', label='beta')
    axes[2].set_title('KL loss + beta schedule')
    axes[2].set_xlabel('Epoch')
    axes[2].legend(loc='upper left')
    ax2.legend(loc='lower right')

    for ax in axes:
        ax.spines[['top','right']].set_visible(False)

    plt.suptitle('CVAE Training Curves', fontsize=12)
    plt.tight_layout()
    path = os.path.join(cfg['output_dir'], 'training_curves.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Loss curves saved → {path}")


# ── Main training loop ────────────────────────────────────────────────────────
def train():
    cfg    = CONFIG
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice           : {device}")
    if device.type == 'cuda':
        print(f"GPU              : {torch.cuda.get_device_name(0)}")
        print(f"VRAM             : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    os.makedirs(cfg['output_dir'], exist_ok=True)
    os.makedirs(cfg['model_dir'],  exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\nLoading data...")
    train_loader, val_loader, _, _ = get_dataloaders(
        cfg['data_dir'], cfg['img_size'], cfg['batch_size']
    )
    print(f"Train batches    : {len(train_loader)}")
    print(f"Val batches      : {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = CVAE(cfg['latent_dim'], cfg['num_classes']).to(device)
    total_params, trainable_params = count_parameters(model)
    print(f"\nTotal parameters : {total_params:,}")
    print(f"Trainable params : {trainable_params:,}")

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimiser = optim.Adam(model.parameters(), lr=cfg['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode='min', factor=0.5, patience=8
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    log_path = os.path.join(cfg['output_dir'], 'training_log.csv')
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch','train_loss','val_loss',
                         'recon_loss','kl_loss','beta','lr','time_s'])

    train_losses, val_losses   = [], []
    recon_losses, kl_losses    = [], []
    betas                      = []
    best_val_loss              = float('inf')
    patience_counter           = 0

    print(f"\nStarting training for {cfg['epochs']} epochs...\n")
    print(f"{'Epoch':>6} {'Train':>10} {'Val':>10} "
          f"{'Recon':>10} {'KL':>10} {'Beta':>8} {'Time':>8}")
    print("-" * 65)

    for epoch in range(1, cfg['epochs'] + 1):
        beta       = get_beta(epoch, cfg)
        start_time = time.time()

        # ── Training phase ────────────────────────────────────────────────────
        model.train()
        epoch_train_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss    = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimiser.zero_grad()
            x_recon, mu, log_var = model(imgs, labels)
            loss, recon, kl      = cvae_loss(x_recon, imgs, mu, log_var, beta)

            loss.backward()
            # gradient clipping — prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

            epoch_train_loss += loss.item()
            epoch_recon_loss += recon.item()
            epoch_kl_loss    += kl.item()

        n_train = len(train_loader)
        epoch_train_loss /= n_train
        epoch_recon_loss /= n_train
        epoch_kl_loss    /= n_train

        # ── Validation phase ──────────────────────────────────────────────────
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                x_recon, mu, log_var = model(imgs, labels)
                loss, _, _           = cvae_loss(x_recon, imgs,
                                                 mu, log_var, beta)
                epoch_val_loss += loss.item()

        epoch_val_loss /= len(val_loader)
        scheduler.step(epoch_val_loss)

        elapsed = time.time() - start_time
        current_lr = optimiser.param_groups[0]['lr']

        # ── Log ───────────────────────────────────────────────────────────────
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        recon_losses.append(epoch_recon_loss)
        kl_losses.append(epoch_kl_loss)
        betas.append(beta)

        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f'{epoch_train_loss:.5f}',
                             f'{epoch_val_loss:.5f}', f'{epoch_recon_loss:.5f}',
                             f'{epoch_kl_loss:.5f}', f'{beta:.6f}',
                             f'{current_lr:.6f}', f'{elapsed:.1f}'])

        print(f"{epoch:>6} {epoch_train_loss:>10.5f} {epoch_val_loss:>10.5f} "
              f"{epoch_recon_loss:>10.5f} {epoch_kl_loss:>10.5f} "
              f"{beta:>8.6f} {elapsed:>6.1f}s")

        # ── Save best model ───────────────────────────────────────────────────
        if epoch_val_loss < best_val_loss:
            best_val_loss    = epoch_val_loss
            patience_counter = 0
            torch.save({
                'epoch'     : epoch,
                'model_state_dict'    : model.state_dict(),
                'optimiser_state_dict': optimiser.state_dict(),
                'val_loss'  : best_val_loss,
                'config'    : cfg,
            }, os.path.join(cfg['model_dir'], 'cvae_best.pth'))
        else:
            patience_counter += 1

        # ── Periodic checkpoint ───────────────────────────────────────────────
        if epoch % cfg['save_every'] == 0:
            torch.save({
                'epoch'     : epoch,
                'model_state_dict'    : model.state_dict(),
                'optimiser_state_dict': optimiser.state_dict(),
                'val_loss'  : epoch_val_loss,
                'config'    : cfg,
            }, os.path.join(cfg['model_dir'], f'cvae_epoch_{epoch:03d}.pth'))

        # ── Reconstruction preview ────────────────────────────────────────────
        if epoch % cfg['preview_every'] == 0 or epoch == 1:
            path = save_reconstruction_preview(
                model, val_loader, epoch, cfg, device
            )
            print(f"  Preview saved → {path}")

        # ── Early stopping ────────────────────────────────────────────────────
        if patience_counter >= cfg['early_stop']:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {cfg['early_stop']} epochs)")
            break

    # ── Final outputs ─────────────────────────────────────────────────────────
    save_loss_curves(train_losses, val_losses,
                     recon_losses, kl_losses, betas, cfg)

    print(f"\nTraining complete.")
    print(f"Best val loss    : {best_val_loss:.5f}")
    print(f"Best model saved : {cfg['model_dir']}/cvae_best.pth")
    print(f"Training log     : {cfg['output_dir']}/training_log.csv")
    print(f"Loss curves      : {cfg['output_dir']}/training_curves.png")
    print(f"Previews         : {cfg['output_dir']}/previews/")


if __name__ == '__main__':
    train()