# src/cgan.py
# Custom Conditional GAN (cGAN) for severity-conditioned medical image generation
#
# Architecture:
#   Generator     : noise + severity label → 128x128 image
#   Discriminator : image + severity label → real/fake score
#
# No custom CUDA kernels — pure PyTorch ops throughout.
# Trained with WGAN-GP loss for training stability on small datasets.

import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_CLASSES  = 3    # mild, moderate, severe
LATENT_DIM   = 128
IMG_SIZE     = 128
IMG_CHANNELS = 3


# ── Utility: embed label as spatial map ──────────────────────────────────────
class LabelEmbedding(nn.Module):
    """
    Converts integer severity label → learned embedding → spatial feature map.
    Injected at every resolution in both G and D.
    """
    def __init__(self, num_classes, embed_dim):
        super().__init__()
        self.embed = nn.Embedding(num_classes, embed_dim)

    def forward(self, labels, h, w):
        e = self.embed(labels)                          # [B, embed_dim]
        return e.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)


# ── Residual block ───────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(x + self.block(x))


# ── Generator ────────────────────────────────────────────────────────────────
class Generator(nn.Module):
    """
    Input  : z [B, LATENT_DIM] + labels [B]
    Output : image [B, 3, 128, 128] in [-1, 1]

    Architecture: FC → reshape 8x8 → 4x upsample blocks → tanh
    Label injected at each resolution via channel concatenation.
    """
    def __init__(self, latent_dim=LATENT_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.latent_dim  = latent_dim
        self.num_classes = num_classes
        embed_dim        = 64

        self.label_embed = LabelEmbedding(num_classes, embed_dim)

        # project z to 8x8 spatial feature map
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512 * 8 * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 8 → 16 (512+64 → 256)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512 + embed_dim, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(256),
        )
        # 16 → 32 (256+64 → 128)
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256 + embed_dim, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(128),
        )
        # 32 → 64 (128+64 → 64)
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128 + embed_dim, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(64),
        )
        # 64 → 128 (64+64 → 3)
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(64 + embed_dim, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, IMG_CHANNELS, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z, labels):
        x = self.fc(z).view(-1, 512, 8, 8)

        e = self.label_embed(labels, 8,  8);  x = self.up1(torch.cat([x, e], dim=1))
        e = self.label_embed(labels, 16, 16); x = self.up2(torch.cat([x, e], dim=1))
        e = self.label_embed(labels, 32, 32); x = self.up3(torch.cat([x, e], dim=1))
        e = self.label_embed(labels, 64, 64); x = self.up4(torch.cat([x, e], dim=1))

        return x   # [B, 3, 128, 128]


# ── Discriminator ─────────────────────────────────────────────────────────────
class Discriminator(nn.Module):
    """
    Input  : image [B, 3, 128, 128] + labels [B]
    Output : scalar score [B, 1] (WGAN — no sigmoid)

    Architecture: 4x downsample blocks → flatten → FC
    Label injected at input via channel concatenation.
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        embed_dim = 64
        self.label_embed = LabelEmbedding(num_classes, embed_dim)

        # input: 3 + embed_dim channels at 128x128
        self.down1 = nn.Sequential(
            nn.Conv2d(IMG_CHANNELS + embed_dim, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64,  128, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Linear(512 * 8 * 8, 1)

    def forward(self, img, labels):
        e = self.label_embed(labels, img.shape[2], img.shape[3])
        x = torch.cat([img, e], dim=1)   # [B, 3+64, 128, 128]

        x = self.down1(x)   # [B, 64,  64, 64]
        x = self.down2(x)   # [B, 128, 32, 32]
        x = self.down3(x)   # [B, 256, 16, 16]
        x = self.down4(x)   # [B, 512,  8,  8]

        return self.fc(x.view(x.shape[0], -1))   # [B, 1]


# ── WGAN-GP gradient penalty ──────────────────────────────────────────────────
def gradient_penalty(D, real, fake, labels, device):
    """
    WGAN-GP gradient penalty.
    Enforces Lipschitz constraint on discriminator without weight clipping.
    Critical for training stability on small datasets.
    """
    B    = real.shape[0]
    eps  = torch.rand(B, 1, 1, 1, device=device)
    interp = (eps * real + (1 - eps) * fake).requires_grad_(True)

    d_interp = D(interp, labels)
    grad = torch.autograd.grad(
        outputs=d_interp,
        inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
    )[0]

    grad_norm = grad.view(B, -1).norm(2, dim=1)
    return ((grad_norm - 1) ** 2).mean()


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    G = Generator().to(device)
    D = Discriminator().to(device)

    z      = torch.randn(4, LATENT_DIM, device=device)
    labels = torch.randint(0, NUM_CLASSES, (4,), device=device)
    imgs   = torch.randn(4, IMG_CHANNELS, IMG_SIZE, IMG_SIZE, device=device)

    fake   = G(z, labels)
    score  = D(fake, labels)
    gp     = gradient_penalty(D, imgs, fake.detach(), labels, device)

    total_g = sum(p.numel() for p in G.parameters())
    total_d = sum(p.numel() for p in D.parameters())

    print(f"Generator output : {fake.shape}  range [{fake.min():.2f}, {fake.max():.2f}]")
    print(f"Discriminator    : {score.shape}")
    print(f"Gradient penalty : {gp.item():.4f}")
    print(f"G parameters     : {total_g:,}")
    print(f"D parameters     : {total_d:,}")
    print("cgan.py sanity check passed.")