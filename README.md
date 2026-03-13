# ML4Sci DeepFalcon GSoC 2026

## Overview
This repository contains solutions for the ML4Sci DeepFalcon GSoC 2026 evaluation tasks.
The dataset consists of Quark/Gluon jet events with 3 detector channels (ECAL, HCAL, Tracks),
each containing 125×125 images.

## Dataset
- **Source**: Quark/Gluon jet events
- **Samples**: 139,306
- **Channels**: ECAL, HCAL, Tracks (125×125 each)
- **Labels**: 0 = Gluon, 1 = Quark (balanced 50/50)
- **Sparsity**: 97-99% zeros per channel

## Tasks Completed

### Common Task 1 — Variational Autoencoder (VAE)
- Trained a VAE on all 3 image channels
- Architecture: Convolutional encoder + Upsampling decoder
- Latent dimension: 64
- Custom loss: MSE + KLD + Sparsity penalty

| Metric | Value |
|--------|-------|
| MSE    | ~0.000000 |
| SSIM   | 0.9996 |
| PSNR   | 78.56 dB |

### Common Task 2 — Jets as Graphs (GNN Classifier)
- Converted jet images → point clouds (non-zero pixels only)
- Built k-NN graphs (k=8) with node features: η, φ, ECAL, HCAL, Tracks
- Model: EdgeConv GNN with global pooling

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 67.47% |
| ROC-AUC   | 0.7376 |
| Precision | 0.6784 |
| Recall    | 0.6379 |

### Specific Task 1 — Graph Autoencoder
- Trained a Graph AE using EdgeConv encoder + MLP decoder
- Latent dimension: 32
- Evaluated with MSE and Wasserstein distance
- Includes latent space visualization (PCA + t-SNE)

| Metric      | Value    |
|-------------|----------|
| MSE         | 0.016870 |
| Wasserstein | 0.028695 |

### Specific Task 3 — Graph Transformer (Generative)
- Implemented a full Transformer encoder-decoder architecture
- Multi-head self-attention on jet point clouds
- Learnable positional encodings
- Near-perfect reconstruction of jet structure

| Metric      | Value    |
|-------------|----------|
| MSE         | 0.000009 |
| Wasserstein | 0.001355 |

## Model Comparison

| Metric      | VAE       | Graph AE | Graph Transformer |
|-------------|-----------|----------|-------------------|
| MSE         | ~0.000000 | 0.016870 | 0.000009          |
| SSIM        | 0.9996    | N/A      | N/A               |
| PSNR        | 78.56 dB  | N/A      | N/A               |
| Wasserstein | N/A       | 0.028695 | 0.001355          |

## Repository Structure
```
ml4sci-falcon/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_VAE.ipynb
│   ├── 03_GNN_Classifier.ipynb
│   ├── 04_Graph_Autoencoder.ipynb
│   └── 05_Graph_Transformer.ipynb
└── results/
    ├── sample_jets.png
    ├── energy_distribution.png
    ├── quark_vs_gluon.png
    ├── vae_side_by_side.png
    ├── vae_loss_curve.png
    ├── gnn_roc_confusion.png
    ├── gnn_loss_acc.png
    ├── graph_ae_latent_space.png
    ├── transformer_reconstructions.png
    └── transformer_latent_space.png

## Setup
pip install -r requirements.txt

## Key Findings
- Gluon jets are wider and softer than quark jets across all 3 detector channels
- Graph-based representations are more physically meaningful than raw images
- Graph Transformer achieves near-perfect reconstruction (MSE=0.000009)
- VAE struggles with extreme sparsity but achieves high SSIM (0.9996)
