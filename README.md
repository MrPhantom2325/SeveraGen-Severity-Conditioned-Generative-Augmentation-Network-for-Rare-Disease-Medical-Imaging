# SeveraGen

**Severity-Conditioned Generative Augmentation Network for Rare Disease Medical Imaging**

SeveraGen is a two-stage generative pipeline that synthesises high-quality, severity-conditioned dermoscopy images to tackle the critical data scarcity problem in rare disease AI diagnostics. It combines a Conditional Variational Autoencoder (CVAE) for learning severity-aware latent distributions with a custom WGAN-GP Conditional GAN for photorealistic image generation — all implemented in pure PyTorch with no custom CUDA kernel dependencies.

---

## The Problem

Medical AI models require thousands of labelled images to generalise, yet rare disease datasets often contain fewer than 100 images per category. This scarcity is worst for the most clinically urgent cases — the severe presentations that matter most for patient safety. Traditional augmentation (flips, rotations, crops) only recombines existing pixels; it cannot synthesise genuinely new examples.

## How SeveraGen Works

The pipeline has three sequential stages:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   STAGE 1       │     │   STAGE 2       │     │   STAGE 3       │
│                 │     │                 │     │                 │
│  Conditional    │────▶│  Conditional    │────▶│  Quality Gate   │
│  VAE            │     │  GAN (WGAN-GP)  │     │  (SSIM + FID)   │
│                 │     │                 │     │                 │
│  Learns per-    │     │  Generates      │     │  Filters out    │
│  severity       │     │  photorealistic │     │  low-quality    │
│  latent         │     │  images from    │     │  images before  │
│  distributions  │     │  learned        │     │  they enter     │
│  (μ, σ²)        │     │  distributions  │     │  training pool  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

The key design principle is **separation of concerns** — the CVAE handles *what* each severity looks like in latent space, while the cGAN handles *rendering* those latent features as realistic dermoscopy images.

## Results

Experiments on the ISIC 2019 skin lesion dataset (Dermatofibroma + Vascular Lesion subset, 492 images):

| Metric | Baseline | CVAE-Aug | cGAN-Aug |
|---|---|---|---|
| Training images | 394 | 1,114 | 1,567 |
| F1 (macro) | 82.69% | 93.53% | **97.97%** |
| F1 — mild | 94.74% | 96.97% | **98.46%** |
| F1 — moderate | 75.56% | 88.37% | **95.45%** |
| F1 — severe | 77.78% | 95.24% | **100.00%** |

The severe class — most clinically critical and most data-scarce (only 30 real images) — went from 77.78% F1 to perfect classification after augmentation with a 17.7x increase in training data.

## Project Structure

```
severagen/
├── src/
│   ├── model.py                 # CVAE architecture (encoder, decoder, reparameterisation)
│   ├── loss.py                  # ELBO loss (MSE reconstruction + β·KL divergence)
│   ├── train.py                 # CVAE training loop with beta annealing
│   ├── cgan.py                  # Generator and Discriminator architectures (WGAN-GP)
│   ├── train_gan.py             # cGAN training loop (5:1 critic ratio)
│   ├── generate.py              # Synthesise images from CVAE-learned distributions
│   ├── quality_gate.py          # SSIM + FID filtering of generated images
│   ├── build_augmented_pool.py  # Assemble real + gated synthetic → training pool
│   ├── classifier.py            # ResNet-18 training and 3-way ablation evaluation
│   ├── dataset.py               # PyTorch Dataset with weighted random sampling
│   ├── label_and_split.py       # Severity proxy labelling (HSV saturation variance)
│   ├── eda.py                   # Exploratory data analysis and report figures
│   ├── visualise.py             # t-SNE, latent interpolation, reconstruction grids
│   ├── verify_dataset.py        # DataLoader sanity check
│   ├── integrity_check.py       # Scan and remove corrupted images
│   ├── pool_sanity_check.py     # Verify augmented pool before classifier training
│   └── dataloader_stress_test.py # Full-epoch stress test
├── data/
│   ├── raw/                     # ISIC 2019 source images and ground truth CSV
│   └── processed/               # Severity-labelled images (mild/moderate/severe)
├── models/                      # Saved checkpoints (cvae_best.pth, cgan_best.pth)
├── outputs/                     # All generated outputs, figures, and results
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10+
- PyTorch 2.0+ with CUDA support
- GPU with ≥16 GB VRAM (tested on NVIDIA A100 MIG 20 GB)

### Installation

```bash
git clone https://github.com/<your-username>/severagen.git
cd severagen
pip install torch torchvision torchmetrics scikit-learn seaborn opencv-python tqdm matplotlib pillow
```

### Data Setup

1. Download the [ISIC 2019 Training Dataset](https://challenge.isic-archive.com/data/#2019) (images + ground truth CSV).
2. Place images in `data/raw/ISIC_2019_Training_Input/` and ground truth in `data/raw/`.
3. Run severity labelling:

```bash
python src/label_and_split.py
```

This filters for rare classes (Dermatofibroma + Vascular Lesion) and assigns severity labels using HSV saturation variance thresholds, producing the `data/processed/{mild,moderate,severe}/` folder structure.

### Running the Pipeline

Execute the stages in order:

```bash
# 1. Verify data integrity
python src/integrity_check.py
python src/verify_dataset.py

# 2. Exploratory data analysis
python src/eda.py

# 3. Train the Conditional VAE (100 epochs)
python src/train.py

# 4. Visualise latent space (t-SNE, interpolations)
python src/visualise.py

# 5. Train the Conditional GAN (300 epochs)
python src/train_gan.py

# 6. Generate synthetic images from learned distributions
python src/generate.py

# 7. Filter with quality gate (SSIM + FID)
python src/quality_gate.py

# 8. Build augmented training pool
python src/build_augmented_pool.py

# 9. Train and evaluate classifiers (3-way ablation)
python src/classifier.py
```

## Architecture Details

### CVAE (Stage 1)

- **Encoder**: 4 conv blocks (32→64→128→256 channels), stride-2 downsampling 128×128→8×8, outputs μ and log σ² (dim 128)
- **Decoder**: Mirrored transposed convolutions, tanh output
- **Conditioning**: Severity one-hot label injected as spatial channel maps at every conv block
- **Training**: 100 epochs, Adam (lr 1e-3), β annealed from 0→0.001 over 30 epochs to prevent posterior collapse

### cGAN (Stage 2)

- **Generator**: FC projection → 4 upsample blocks with residual connections and label embedding injection at every resolution
- **Discriminator**: 4 downsample blocks with instance normalisation, scalar WGAN output (no sigmoid)
- **Training**: 300 epochs, WGAN-GP (λ=10), n_critic=5, separate Adam optimisers (G: 1e-4, D: 4e-4)
- **Pure PyTorch**: No custom CUDA kernels — runs on any CUDA-capable environment

### Quality Gate (Stage 3)

- **SSIM filter**: Per-image nearest-neighbour SSIM against real references (thresholds: mild/moderate 0.35, severe 0.20)
- **FID evaluation**: Per-class distributional similarity via Inception-v3 features
- **Result**: 97.7% overall acceptance rate (1,466 of 1,500 generated images passed)

## Key Design Decisions

**Why WGAN-GP over standard GAN?** Standard GAN loss suffers from vanishing gradients and mode collapse on small datasets. Wasserstein distance provides smoother gradients throughout training and the gradient penalty prevents discriminator overfitting to limited real examples.

**Why two stages instead of just a GAN?** The CVAE explicitly models per-severity latent distributions, giving the cGAN generator structured, severity-aware noise to work with. This produces better class separation than conditioning the GAN on labels alone.

**Why calibrated SSIM thresholds?** With only 30 real severe images, a uniform threshold would reject disproportionately many severe images (fewer references → lower nearest-neighbour SSIM). The relaxed threshold for severe (0.20 vs 0.35) compensates for the smaller reference set.

## Citation

If you use SeveraGen in your research, please cite:

```bibtex
@misc{severagen2025,
  title={SeveraGen: Severity-Conditioned Generative Augmentation Network},
  year={2025},
  note={Academic project report}
}
```

## Acknowledgements

- [ISIC 2019 Challenge](https://challenge.isic-archive.com/data/#2019) for the dermoscopy dataset
- WGAN-GP formulation by [Gulrajani et al. (2017)](https://arxiv.org/abs/1704.00028)
- β-VAE framework by [Higgins et al. (2017)](https://openreview.net/forum?id=Sy2fzU9gl)

## License

This project was developed as an academic submission. Please contact the authors for licensing information.
