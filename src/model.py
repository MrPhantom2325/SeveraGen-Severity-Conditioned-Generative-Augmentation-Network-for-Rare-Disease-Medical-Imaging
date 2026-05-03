# src/model.py
# Severity-Conditioned Variational Autoencoder (CVAE)
#
# Architecture overview:
#   Encoder : Image (3x128x128) + severity label → mu, log_var (latent_dim)
#   Reparameterisation : z = mu + eps * exp(0.5 * log_var)
#   Decoder : z + severity label → reconstructed image (3x128x128)
#
# Conditioning: severity label is one-hot encoded and injected at
#   - encoder input  (concatenated as extra channels)
#   - decoder input  (concatenated to latent vector z)
# This forces separate latent clusters per severity class.

import torch
import torch.nn as nn


NUM_CLASSES = 3     # mild, moderate, severe
LATENT_DIM  = 128   # dimensionality of the latent space z


# ── Utility: one-hot → spatial map ──────────────────────────────────────────
def label_to_spatial(labels, num_classes, h, w, device):
    """
    Converts integer severity labels to a spatial one-hot tensor
    that can be concatenated channel-wise with feature maps.

    Example:
        labels = [0, 2, 1]   (mild, severe, moderate)
        output shape = [3, num_classes, h, w]
        Each channel is all-1s for the matching class, 0s elsewhere.
    """
    B = labels.shape[0]
    one_hot = torch.zeros(B, num_classes, device=device)
    one_hot.scatter_(1, labels.unsqueeze(1), 1.0)          # [B, num_classes]
    # expand to spatial: [B, num_classes, h, w]
    return one_hot.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)


# ── Encoder ──────────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    """
    CNN encoder that compresses (image + label) into latent parameters.

    Input  : [B, 3 + num_classes, 128, 128]
               (image channels + one-hot severity channels)
    Output : mu [B, latent_dim], log_var [B, latent_dim]

    Architecture:
        Conv block 1 : 128x128 → 64x64   (6→32  channels)
        Conv block 2 : 64x64  → 32x32   (32→64  channels)
        Conv block 3 : 32x32  → 16x16   (64→128 channels)
        Conv block 4 : 16x16  → 8x8     (128→256 channels)
        Flatten + FC → mu, log_var
    """

    def __init__(self, latent_dim=LATENT_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.latent_dim  = latent_dim
        self.num_classes = num_classes

        # input channels = 3 (RGB) + num_classes (one-hot label)
        in_ch = 3 + num_classes

        self.encoder = nn.Sequential(
            # block 1: 128→64
            nn.Conv2d(in_ch, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # block 2: 64→32
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # block 3: 32→16
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # block 4: 16→8
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # flattened size = 256 * 8 * 8
        self.flat_dim = 256 * 8 * 8

        self.fc_mu      = nn.Linear(self.flat_dim, latent_dim)
        self.fc_log_var = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x, labels):
        """
        Args:
            x      : [B, 3, 128, 128]  image tensor, normalised [-1,1]
            labels : [B]               integer severity labels
        Returns:
            mu      : [B, latent_dim]
            log_var : [B, latent_dim]
        """
        B, _, H, W = x.shape
        # concatenate label as spatial channels
        label_map = label_to_spatial(labels, self.num_classes, H, W, x.device)
        x_cond    = torch.cat([x, label_map], dim=1)   # [B, 6, 128, 128]

        h = self.encoder(x_cond)                        # [B, 256, 8, 8]
        h = h.view(B, -1)                               # [B, 16384]

        mu      = self.fc_mu(h)                         # [B, latent_dim]
        log_var = self.fc_log_var(h)                    # [B, latent_dim]

        return mu, log_var


# ── Decoder ──────────────────────────────────────────────────────────────────
class Decoder(nn.Module):
    """
    CNN decoder that reconstructs an image from (z + label).

    Input  : z [B, latent_dim] + label [B, num_classes]
               → concatenated → [B, latent_dim + num_classes]
    Output : [B, 3, 128, 128]  reconstructed image, tanh → [-1,1]

    Architecture:
        FC → 256*8*8
        Reshape [B, 256, 8, 8]
        TransposedConv 1 : 8→16    (256→128)
        TransposedConv 2 : 16→32   (128→64)
        TransposedConv 3 : 32→64   (64→32)
        TransposedConv 4 : 64→128  (32→3)
        Tanh → output in [-1,1]
    """

    def __init__(self, latent_dim=LATENT_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.latent_dim  = latent_dim
        self.num_classes = num_classes
        self.flat_dim    = 256 * 8 * 8

        # project (z + one-hot label) up to spatial feature map
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + num_classes, self.flat_dim),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            # block 1: 8→16
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # block 2: 16→32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # block 3: 32→64
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # block 4: 64→128
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()   # output in [-1, 1]
        )

    def forward(self, z, labels):
        """
        Args:
            z      : [B, latent_dim]
            labels : [B]   integer severity labels
        Returns:
            x_recon : [B, 3, 128, 128]
        """
        # one-hot encode label
        one_hot = torch.zeros(z.shape[0], self.num_classes, device=z.device)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

        z_cond = torch.cat([z, one_hot], dim=1)     # [B, latent_dim+3]
        h      = self.fc(z_cond)                    # [B, 256*8*8]
        h      = h.view(-1, 256, 8, 8)              # [B, 256, 8, 8]

        return self.decoder(h)                      # [B, 3, 128, 128]


# ── CVAE (full model) ─────────────────────────────────────────────────────────
class CVAE(nn.Module):
    """
    Full Conditional VAE combining Encoder + Decoder.
    Exposes encode(), decode(), reparameterise(), and forward().
    """

    def __init__(self, latent_dim=LATENT_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.encoder = Encoder(latent_dim, num_classes)
        self.decoder = Decoder(latent_dim, num_classes)
        self.latent_dim  = latent_dim
        self.num_classes = num_classes

    def reparameterise(self, mu, log_var):
        """
        Reparameterisation trick:
            z = mu + epsilon * sigma    where epsilon ~ N(0, I)

        During training  : samples stochastically (enables backprop through z)
        During inference : returns mu directly (deterministic, best estimate)

        This trick is the key innovation of VAEs — it allows gradients to flow
        through the sampling operation by separating the randomness (epsilon)
        from the learned parameters (mu, log_var).
        """
        if self.training:
            std = torch.exp(0.5 * log_var)     # sigma = exp(log_var / 2)
            eps = torch.randn_like(std)        # epsilon ~ N(0, I)
            return mu + eps * std
        else:
            return mu   # deterministic at eval time

    def encode(self, x, labels):
        mu, log_var = self.encoder(x, labels)
        z = self.reparameterise(mu, log_var)
        return z, mu, log_var

    def decode(self, z, labels):
        return self.decoder(z, labels)

    def forward(self, x, labels):
        """
        Full forward pass: encode → reparameterise → decode

        Returns:
            x_recon : [B, 3, 128, 128]  reconstructed image
            mu      : [B, latent_dim]   latent mean
            log_var : [B, latent_dim]   latent log variance
        """
        mu, log_var = self.encoder(x, labels)
        z           = self.reparameterise(mu, log_var)
        x_recon     = self.decoder(z, labels)
        return x_recon, mu, log_var


# ── Parameter count utility ───────────────────────────────────────────────────
def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = CVAE(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES).to(device)

    # dummy batch
    B      = 8
    x      = torch.randn(B, 3, 128, 128).to(device)
    labels = torch.randint(0, NUM_CLASSES, (B,)).to(device)

    # forward pass
    x_recon, mu, log_var = model(x, labels)

    print(f"\nInput shape       : {x.shape}")
    print(f"Reconstructed     : {x_recon.shape}")
    print(f"mu shape          : {mu.shape}")
    print(f"log_var shape     : {log_var.shape}")
    print(f"Output range      : [{x_recon.min():.3f}, {x_recon.max():.3f}]")

    total, trainable = count_parameters(model)
    print(f"\nTotal parameters  : {total:,}")
    print(f"Trainable params  : {trainable:,}")

    # test encode / decode separately
    z, mu, log_var = model.encode(x, labels)
    x_decoded      = model.decode(z, labels)
    print(f"\nEncode → z shape  : {z.shape}")
    print(f"Decode → img shape: {x_decoded.shape}")

    print("\nmodel.py sanity check passed.")