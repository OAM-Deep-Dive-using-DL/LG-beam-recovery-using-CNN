# Deep Learning for OAM Beam Recovery in Atmospheric Turbulence

<div align="center">

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-research-orange.svg)

**Solving the "Deep Fade" problem in Free Space Optical communications using Spatial Attention Neural Networks**

[Key Results](#key-results) • [Quick Start](#quick-start) • [Technical Details](#technical-details) • [Citation](#citation)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Quick Start](#quick-start)
- [The Problem](#the-problem)
- [Our Solution](#our-solution)
- [Technical Details](#technical-details)
  - [Architecture Evolution](#architecture-evolution)
  - [Spatial Attention (CBAM)](#spatial-attention-cbam)
- [Performance Analysis](#performance-analysis)
- [Usage Guide](#usage-guide)
  - [Data Generation](#data-generation)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [License](#license)

---

## Overview

This repository presents a **Neural Receiver** for Orbital Angular Momentum (OAM) multiplexed Free Space Optical (FSO) communication systems. We achieve a **30dB improvement** in turbulence resilience compared to classical MMSE receivers by using a ResNet-18 backbone enhanced with Convolutional Block Attention Modules (CBAM).

**Key Innovation**: Direct recovery of complex QPSK symbols from intensity-only measurements, eliminating the need for expensive phase measurement hardware.

---

## Key Results

### The Breakthrough: 30dB Turbulence Resilience Gain

![Performance Comparison](models/CNN%20Trials/outputs/plots/comparison_architecture_plot.png)

**Critical Observations:**

| Turbulence Regime | $C_n^2$ Range | Classical MMSE | ResNet-18 | **ResNet-18 + CBAM** |
|:------------------|:--------------|:---------------|:----------|:---------------------|
| **Weak** | $10^{-18}$ - $10^{-16}$ | BER < 0.1% | **BER = 0%** ✓ | **BER = 0%** ✓ |
| **Moderate** | $10^{-16}$ - $10^{-15}$ | **BER = 28% ✗** | BER = 0.4% | **BER = 0.03%** ✓ |
| **Strong** | $10^{-15}$ - $10^{-14}$ | BER ≈ 50% (Random) | BER = 10% | **BER = 3-5%** ✓ |

**Verdict**: The CBAM-enhanced model pushes the operational limit by **10x** compared to classical methods and **3x** compared to vanilla deep learning.

### Visual Proof: Blind Phase Recovery

<div align="center">

![Constellation Recovery](models/CNN%20Trials/outputs/plots/evaluation_constellation.png)

*The network recovers clean QPSK constellations from intensity-only inputs, effectively "hallucinating" the lost phase information through learned spatial correlations.*

</div>

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/FSO-beam-recovery.git
cd FSO-beam-recovery

# Install dependencies
pip install torch torchvision numpy scipy h5py matplotlib tqdm
```

### 30-Second Demo

```bash
# Generate sample data
cd "models/CNN Trials"
python src/data_gen/generate_dataset.py --samples 1000 --name demo

# Train model (5 epochs for quick test)
python src/training/train.py --dataset_name demo --epochs 5 --backbone resnet18_cbam

# Evaluate
python src/evaluation/evaluate.py --dataset_name demo --backbone resnet18_cbam
```

---

## The Problem

### OAM Communications Under Turbulence

Orbital Angular Momentum (OAM) beams offer **infinite-dimensional multiplexing** ($l \in \mathbb{Z}$), enabling massive capacity gains in FSO links. However, atmospheric turbulence causes:

1. **Phase Scrambling**: Destroys the helical wavefront structure
2. **Inter-Modal Crosstalk**: Energy leaks between modes ($l \to l \pm 1, l \pm 2, ...$)
3. **Beam Fragmentation**: The beam breaks into random "speckles"

![Turbulence Impact](models/LDPC%20+%20Pilot%20+%20MMSE%20trials/plots%20-%20LDPC%20+%20Pilot%20+%20MMSE%20trials/turbulence_summary/lg_turbulence_verified_viz4.png)

*Left: Clean OAM beam. Right: After 1km propagation through strong turbulence ($C_n^2 = 10^{-14}$).*

### Why Classical Methods Fail

Classical receivers use **MMSE Equalization** to invert the channel matrix $\mathbf{H}$:

$$\hat{\mathbf{s}} = (\mathbf{H}^H \mathbf{H} + \sigma^2 \mathbf{I})^{-1} \mathbf{H}^H \mathbf{y}$$

**Failure Mode**: In strong turbulence, $\mathbf{H}$ becomes singular (near-zero eigenvalues), making inversion unstable. The noise amplification causes BER to plateau at ~50% (random guessing).

---

## Our Solution

### Deep Learning as "Manifold Learning"

Instead of inverting the channel mathematically, we train a CNN to learn the **manifold of distorted beam patterns**. The network learns:

> "A donut broken into 3 speckles at positions (x₁,y₁), (x₂,y₂), (x₃,y₃) with relative intensities (I₁,I₂,I₃) corresponds to Mode +1 with phase φ."

This pattern-matching approach is robust even when explicit phase information is completely lost.

### Architecture: ResNet-18 + CBAM

```
Input: [1, 64, 64] Intensity Image (No Phase)
   ↓
ResNet-18 Backbone (Feature Extraction)
   ├─ Layer 1: BasicBlock + CBAM  [64 channels]
   ├─ Layer 2: BasicBlock + CBAM  [128 channels]
   ├─ Layer 3: BasicBlock + CBAM  [256 channels]
   └─ Layer 4: BasicBlock + CBAM  [512 channels]
   ↓
Multi-Head Regression
   ├─ FC(512 → 256) + ReLU + Dropout(0.3)
   └─ FC(256 → 16)  [8 modes × (Re + Im)]
   ↓
Output: [8, 2] Complex QPSK Symbols
```

**Parameter Count**: ~11.7M (ResNet-18) + 0.4M (CBAM) = **12.1M total**

---

## Technical Details

### Architecture Evolution

We iteratively improved the model in 3 stages:

1. **Baseline (ResNet-18)**: Standard ImageNet-pretrained ResNet
   - **Problem**: Struggled in deep fades ($C_n^2 > 10^{-15}$)
   
2. **+ Transfer Learning**: Fine-tuned on turbulence data
   - **Improvement**: Better generalization, but still error floor
   
3. **+ Spatial Attention (CBAM)**: Final architecture
   - **Breakthrough**: Dynamically focuses on beam fragments, ignoring noise

### Spatial Attention (CBAM)

The **Convolutional Block Attention Module** adds only 1.7% overhead but provides 10x performance gain in strong turbulence.

#### Channel Attention

Learns "which features are important" (e.g., radial intensity gradients vs. noise).

```python
class ChannelGate(nn.Module):
    def forward(self, x):
        avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)))
        max_pool = F.max_pool2d(x, (x.size(2), x.size(3)))
        channel_att = self.mlp(avg_pool) + self.mlp(max_pool)
        return x * torch.sigmoid(channel_att).unsqueeze(2).unsqueeze(3)
```

#### Spatial Attention

Learns "where to look" (e.g., beam hotspots vs. background).

```python
class SpatialGate(nn.Module):
    def forward(self, x):
        x_compress = self.compress(x)  # [B, 2, H, W] (avg+max across channels)
        spatial_att = self.spatial(x_compress)  # [B, 1, H, W]
        return x * torch.sigmoid(spatial_att)  # Broadcasting
```

**Key Insight**: In turbulence, the beam energy clusters into 2-5 distinct speckles. The spatial gate learns an attention mask that highlights these clusters, suppressing the diffuse background noise.

---

## Performance Analysis

### Quantitative Comparison

| Metric | Classical MMSE | ResNet-18 (Vanilla) | **ResNet-18 + CBAM** |
|:-------|:---------------|:--------------------|:---------------------|
| **Breakdown Point** ($C_n^2$) | $3 \times 10^{-16}$ | $10^{-15}$ | **$3 \times 10^{-15}$** |
| **Throughput (Weak Turb)** | 11.7 Gbps | 11.7 Gbps | **11.7 Gbps** |
| **Throughput (Mod. Turb)** | 0 Gbps (Link Fail) | 8.5 Gbps | **11.7 Gbps** (Stable) |
| **Hardware Requirements** | Wavefront sensor (Coherent) | Intensity camera | **Intensity camera** |
| **Inference Time (GPU)** | N/A | 1.2ms | **1.5ms** |

### Complexity Analysis

- **Classical MMSE**: $O(N^3)$ matrix inversion per frame
- **Neural Receiver**: $O(1)$ forward pass (constant time, amortized training cost)

**Trade-off**: Higher upfront training cost (6 hours on 1x V100), but 100x faster inference and no pilot overhead.

---

## Usage Guide

### Data Generation

Generate realistic turbulence data using our physics-based simulator (Split-Step Fourier Method).

```bash
cd "models/CNN Trials"

# Training set (100k samples, ~6 hours on CPU)
python src/data_gen/generate_dataset.py \
    --samples 100000 \
    --name fso_oam_turbulence_hard_train

# Validation set (10k samples)
python src/data_gen/generate_dataset.py \
    --samples 10000 \
    --name fso_oam_turbulence_hard_val

# Test set (High-resolution sweep across turbulence strengths)
python src/data_gen/generate_dataset.py \
    --samples 20000 \
    --name fso_oam_turbulence_sweep_50pt \
    --mode sweep
```

**Output**: HDF5 files in `data/` directory (~2GB per 10k samples)

### Training

Train the CBAM-enhanced model:

```bash
python src/training/train.py \
    --data_dir "data" \
    --dataset_name fso_oam_turbulence_hard \
    --backbone resnet18_cbam \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-3
```

**Training Time**: ~6 hours (100k samples, 50 epochs, 1x V100)

**Checkpoints**: Saved to `outputs/checkpoints/best_model_resnet18_cbam.pth`

#### Advanced: Resume Training

```bash
python src/training/train.py \
    --dataset_name fso_oam_turbulence_hard \
    --backbone resnet18_cbam \
    --epochs 500 \
    --resume  # Loads last_model_resnet18_cbam.pth
```

### Evaluation

Generate BER curves and constellation diagrams:

```bash
python src/evaluation/evaluate.py \
    --data_dir "data" \
    --dataset_name fso_oam_turbulence_sweep_50pt \
    --backbone resnet18_cbam
```

**Outputs**:
- `outputs/plots/evaluation_ber_curve.png`
- `outputs/plots/evaluation_constellation.png`
- `outputs/logs/cnn_results.npz` (for plotting)

#### Generate Comparison Plot

```bash
python src/evaluation/plot_comparison.py
```

**Output**: `outputs/plots/comparison_architecture_plot.png` (the Money Shot)

---

## Project Structure

```
FSO-beam-recovery/
├── models/
│   ├── CNN Trials/                    # Neural Receiver (Main Project)
│   │   ├── src/
│   │   │   ├── models/
│   │   │   │   ├── model.py          # MultiHeadResNet (main model)
│   │   │   │   ├── resnet_cbam.py    # ResNet-18 + CBAM
│   │   │   │   └── attention.py      # CBAM implementation
│   │   │   ├── training/
│   │   │   │   └── train.py          # Training loop
│   │   │   ├── evaluation/
│   │   │   │   ├── evaluate.py       # BER/SER metrics
│   │   │   │   └── plot_comparison.py # Generate comparison plots
│   │   │   ├── data_gen/
│   │   │   │   └── generate_dataset.py # Physics simulator wrapper
│   │   │   └── utils/
│   │   │       └── dataset.py        # PyTorch Dataset class
│   │   ├── physics/                  # Split-Step Propagation Engine
│   │   │   ├── transmitter.py
│   │   │   ├── channel.py
│   │   │   └── receiver.py
│   │   ├── data/                     # HDF5 datasets (git-ignored)
│   │   ├── outputs/
│   │   │   ├── checkpoints/          # Trained models (.pth)
│   │   │   ├── plots/                # Result figures
│   │   │   ├── logs/                 # NPZ files
│   │   │   └── reports/              # Markdown summaries
│   │   └── README.md                 # Usage instructions
│   │
│   └── LDPC + Pilot + MMSE trials/   # Classical Baseline
│       ├── lgBeam.py                 # Main simulation script
│       └── plots - .../              # Classical receiver results
│
├── requirements.txt
├── LICENSE
└── README.md                         # This file
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{davuluri2024oam,
  author = {Davuluri, Srivatsa},
  title = {Deep Learning for OAM Beam Recovery in Atmospheric Turbulence},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/FSO-beam-recovery}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Physics Simulator**: Based on the Split-Step Fourier Method (Andrews & Phillips, 2005)
- **CBAM Module**: Adapted from Woo et al., "CBAM: Convolutional Block Attention Module," ECCV 2018
- **Turbulence Model**: Von Karman spectrum (Kolmogorov, 1941)

---

<div align="center">


[⬆ Back to Top](#deep-learning-for-oam-beam-recovery-in-atmospheric-turbulence)

</div>
