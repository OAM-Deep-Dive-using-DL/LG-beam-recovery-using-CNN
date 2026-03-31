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

## Setup

```bash
cd "/Users/srivatsadavuluri/Developer/Wireless Communications Related/FSO beam recovery"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For manuscript builds, ensure `pdflatex` and `bibtex` are installed.

---

## Classical Baseline (MMSE/ZF) Sweep

```bash
cd "models/LDPC + Pilot + MMSE trials"
python sweep_baseline.py \
  --cn2-min 1e-18 \
  --cn2-max 1e-12 \
  --num-points 41 \
  --repeats 3 \
  --equalizers mmse zf
```

Default output folder: `ieee_cn2_sweep_results/`  
Main artifacts:

- `baseline_sweep_raw.json`
- `baseline_sweep_aggregated.json`
- `cn2_vs_ber.png/.pdf`
- `pre_post_ldpc_ber.png/.pdf`

---

## Neural Dataset Generation

Generator:

- `models/CNN Trials/data/generators/generate_dataset.py`

Configs:

- `models/CNN Trials/data/configs/config.json`
- `models/CNN Trials/data/configs/curriculum_lvl1_ideal.json` ... `curriculum_lvl5_extreme.json`

Generate one config:

```bash
cd "models/CNN Trials/data/generators"
python generate_dataset.py --config configs/config.json --split all
```

Generate full curriculum configs:

```bash
cd "models/CNN Trials/data/generators"
python generate_dataset.py \
  --config configs/curriculum_lvl1_ideal.json \
           configs/curriculum_lvl2_weak.json \
           configs/curriculum_lvl3_moderate.json \
           configs/curriculum_lvl4_strong.json \
           configs/curriculum_lvl5_extreme.json \
  --split all
```

---

## Neural Training

Script:

- `models/CNN Trials/src/training/train.py`

Example:

```bash
cd "models/CNN Trials/src/training"
python train.py \
  --data_dir ../../data/generated_curriculum \
  --dataset_name fso_oam_turbulence_v1 \
  --backbone convnext_tiny \
  --epochs 150 \
  --batch_size 32 \
  --loss polar \
  --device auto
```

Curriculum runner:

```bash
cd "models/CNN Trials"
python src/training/train_curriculum.py
```

---

## Neural Evaluation

Script:

- `models/CNN Trials/src/evaluation/evaluate.py`

```bash
cd "models/CNN Trials/src/evaluation"
python evaluate.py \
  --data_dir ../../data/generated_curriculum \
  --dataset_name fso_oam_turbulence_v1 \
  --backbone convnext_tiny \
  --device auto
```

Outputs include BER-vs-`C_n^2` plots, constellation plots, and `.npz` result arrays.

---

## Manuscript Build

Primary working source:

- `Manuscript/manuscript-2.tex`

```bash
cd Manuscript
pdflatex manuscript-2.tex
bibtex manuscript-2
pdflatex manuscript-2.tex
pdflatex manuscript-2.tex
```

Other variants:

- `Manuscript/manuscript.tex`
- `Manuscript/bare.tex`

---

## Citation

If you use this repository, cite your manuscript/software record for:

**OAM-Assisted High-Capacity Transmission: A Link-Level Performance Study**

---

## License

MIT License. See `LICENSE`.
# OAM-Assisted High-Capacity Transmission: A Link-Level Performance Study

This repository contains an end-to-end simulation and evaluation framework for OAM-multiplexed optical wireless links under atmospheric turbulence. It includes:

- a classical coherent baseline (pilot-aided LS + MMSE/ZF + LDPC),
- an intensity-only neural receiver (ConvNeXt/EfficientNet variants),
- a physics-grounded dataset generator (SSFM + Kolmogorov phase screens),
- and IEEE-style manuscript assets.

The project is organized to support reproducible, matched-condition comparisons between classical and neural receivers across a controlled `C_n^2` sweep.

---

## Repository Layout

- `models/LDPC + Pilot + MMSE trials/`  
  Classical link-level baseline (`pipeline.py`, `sweep_baseline.py`, BER/channel-matrix outputs).

- `models/CNN Trials/`  
  Neural pipeline:
  - `physics/`
  - `data/generators/`
  - `src/training/`
  - `src/evaluation/`

- `Manuscript/`  
  Paper sources (`manuscript.tex`, `manuscript-2.tex`, `references.bib`) and figure assets.

- `requirements.txt`  
  Python dependencies.

---

## Environment Setup

```bash
cd "/Users/srivatsadavuluri/Developer/Wireless Communications Related/FSO beam recovery"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For LaTeX builds, ensure `pdflatex` and `bibtex` are available.

---

## Classical Baseline Workflow

### Run canonical `C_n^2` sweep

```bash
cd "models/LDPC + Pilot + MMSE trials"
python sweep_baseline.py \
  --cn2-min 1e-18 \
  --cn2-max 1e-12 \
  --num-points 41 \
  --repeats 3 \
  --equalizers mmse zf
```

Key outputs (default: `ieee_cn2_sweep_results/`):

- `baseline_sweep_raw.json`
- `baseline_sweep_aggregated.json`
- `cn2_vs_ber.png/.pdf`
- `pre_post_ldpc_ber.png/.pdf`
- representative channel-matrix image

---

## Neural Dataset Generation

Generator entrypoint:

- `models/CNN Trials/data/generators/generate_dataset.py`

Available configs:

- `models/CNN Trials/data/configs/config.json`
- `models/CNN Trials/data/configs/curriculum_lvl1_ideal.json` ... `curriculum_lvl5_extreme.json`

### Generate one config (all splits)

```bash
cd "models/CNN Trials/data/generators"
python generate_dataset.py --config configs/config.json --split all
```

### Generate curriculum datasets

```bash
cd "models/CNN Trials/data/generators"
python generate_dataset.py \
  --config configs/curriculum_lvl1_ideal.json \
           configs/curriculum_lvl2_weak.json \
           configs/curriculum_lvl3_moderate.json \
           configs/curriculum_lvl4_strong.json \
           configs/curriculum_lvl5_extreme.json \
  --split all
```

By default, `.h5` outputs are written under `models/CNN Trials/data/` (or a custom `--output-dir`).

---

## Neural Training

Training script:

- `models/CNN Trials/src/training/train.py`

```bash
cd "models/CNN Trials/src/training"
python train.py \
  --data_dir ../../data/generated_curriculum \
  --dataset_name fso_oam_turbulence_v1 \
  --backbone convnext_tiny \
  --epochs 150 \
  --batch_size 32 \
  --loss polar \
  --device auto
```

Curriculum helper:

```bash
cd "models/CNN Trials"
python src/training/train_curriculum.py
```

---

## Neural Evaluation

Evaluation script:

- `models/CNN Trials/src/evaluation/evaluate.py`

```bash
cd "models/CNN Trials/src/evaluation"
python evaluate.py \
  --data_dir ../../data/generated_curriculum \
  --dataset_name fso_oam_turbulence_v1 \
  --backbone convnext_tiny \
  --device auto
```

Typical outputs:

- BER-vs-`C_n^2` plot (`.png` + `.pdf`)
- constellation plot (`.png` + `.pdf`)
- result arrays (`.npz`)

---

## Manuscript Build

Main working manuscript:

- `Manuscript/manuscript-2.tex`

```bash
cd Manuscript
pdflatex manuscript-2.tex
bibtex manuscript-2
pdflatex manuscript-2.tex
pdflatex manuscript-2.tex
```

Other manuscript variants:

- `Manuscript/manuscript.tex`
- `Manuscript/bare.tex`

---

## Notes

- Baseline-first methodology with matched channel conditions across classical and neural branches.
- Neural branch is intensity-only and compared against coherent baseline under the same physics.
- Figure assets include both raster and TikZ/LaTeX publication-quality versions.

---

## Citation

If you use this repository, cite your manuscript/software record for:

**OAM-Assisted High-Capacity Transmission: A Link-Level Performance Study**

---

## License

This project is licensed under the MIT License. See `LICENSE`.
# OAM-Assisted High-Capacity Transmission: A Link-Level Performance Study

This repository contains an end-to-end simulation and evaluation framework for OAM-multiplexed optical wireless links under atmospheric turbulence. It includes:

- a **classical coherent baseline** (pilot-aided LS + MMSE/ZF + LDPC),
- an **intensity-only neural receiver** (ConvNeXt/EfficientNet variants),
- a **physics-grounded dataset generator** (SSFM + Kolmogorov phase screens),
- and the full **IEEE-style manuscript assets**.

The project is organized to support reproducible, matched-condition comparisons between classical and neural receivers across a controlled `C_n^2` sweep.

---

## Repository Layout

- `models/LDPC + Pilot + MMSE trials/`  
  Classical link-level baseline (`pipeline.py`, `sweep_baseline.py`, channel matrix/BER outputs).

- `models/CNN Trials/`  
  Neural pipeline:
  - physics modules (`physics/`),
  - dataset generation (`data/generators/`),
  - training (`src/training/`),
  - evaluation and comparison plots (`src/evaluation/`).

- `Manuscript/`  
  Paper sources (`manuscript.tex`, `manuscript-2.tex`, `references.bib`) and figure assets.

- `requirements.txt`  
  Python dependencies.

---

## Environment Setup

```bash
cd "/Users/srivatsadavuluri/Developer/Wireless Communications Related/FSO beam recovery"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For LaTeX builds, ensure a TeX distribution is installed (e.g., TeX Live / MacTeX with `pdflatex` and `bibtex`).

---

## 1) Classical Baseline Workflow

### Run canonical `C_n^2` sweep

```bash
cd "models/LDPC + Pilot + MMSE trials"
python sweep_baseline.py \
  --cn2-min 1e-18 \
  --cn2-max 1e-12 \
  --num-points 41 \
  --repeats 3 \
  --equalizers mmse zf
```

Key outputs (inside `ieee_cn2_sweep_results/` by default):

- `baseline_sweep_raw.json`
- `baseline_sweep_aggregated.json`
- `cn2_vs_ber.png/.pdf`
- `pre_post_ldpc_ber.png/.pdf`
- representative channel-matrix image

---

## 2) Neural Dataset Generation

Dataset generator entrypoint:

- `models/CNN Trials/data/generators/generate_dataset.py`

Available configs:

- `models/CNN Trials/data/configs/config.json`
- `models/CNN Trials/data/configs/curriculum_lvl1_ideal.json` ... `curriculum_lvl5_extreme.json`

### Generate one full split set from a config

```bash
cd "models/CNN Trials/data/generators"
python generate_dataset.py --config configs/config.json --split all
```

### Generate curriculum-stage datasets

```bash
cd "models/CNN Trials/data/generators"
python generate_dataset.py \
  --config configs/curriculum_lvl1_ideal.json \
           configs/curriculum_lvl2_weak.json \
           configs/curriculum_lvl3_moderate.json \
           configs/curriculum_lvl4_strong.json \
           configs/curriculum_lvl5_extreme.json \
  --split all
```

By default, `.h5` outputs are written under `models/CNN Trials/data/` (or a custom `--output-dir`).

---

## 3) Neural Training

Training script:

- `models/CNN Trials/src/training/train.py`

### Standard training

```bash
cd "models/CNN Trials/src/training"
python train.py \
  --data_dir ../../data/generated_curriculum \
  --dataset_name fso_oam_turbulence_v1 \
  --backbone convnext_tiny \
  --epochs 150 \
  --batch_size 32 \
  --loss polar \
  --device auto
```

### Curriculum progression helper

```bash
cd "models/CNN Trials"
python src/training/train_curriculum.py
```

This script runs staged training (`ideal -> weak -> moderate -> strong -> extreme`) with checkpoint carryover.

---

## 4) Neural Evaluation

Evaluation script:

- `models/CNN Trials/src/evaluation/evaluate.py`

```bash
cd "models/CNN Trials/src/evaluation"
python evaluate.py \
  --data_dir ../../data/generated_curriculum \
  --dataset_name fso_oam_turbulence_v1 \
  --backbone convnext_tiny \
  --device auto
```

Typical outputs:

- BER-vs-`C_n^2` curve (`.png` + `.pdf`)
- constellation plot (`.png` + `.pdf`)
- compressed result arrays (`.npz`)

---

## 5) Manuscript Build

Main working paper (with controlled float pass and iterative figure placement):

- `Manuscript/manuscript-2.tex`

```bash
cd Manuscript
pdflatex manuscript-2.tex
bibtex manuscript-2
pdflatex manuscript-2.tex
pdflatex manuscript-2.tex
```

Other manuscript variants:

- `Manuscript/manuscript.tex`
- `Manuscript/bare.tex`

---

## Notes on Current Scope

- The repository currently reflects a **baseline-first** methodology: same turbulence/channel backbone used for both classical and neural branches.
- Neural receiver is **intensity-only** and compared against coherent baseline under matched simulation conditions.
- Figure assets include both raster images and LaTeX/TikZ reconstructions for publication-quality rendering.

---

## Citation

If you use this repository in research, please cite your manuscript/software record for:

**OAM-Assisted High-Capacity Transmission: A Link-Level Performance Study**

(Add your final BibTeX entry here once publication metadata is fixed.)

---

## License

This project is licensed under the MIT License. See `LICENSE`.
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

## Citation

If you use this code in your research, please cite:

```bibtex
@software{davuluri2024oam,
  author = {Davuluri, Srivatsa},
  title = {Deep Learning for OAM Beam Recovery in Atmospheric Turbulence},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/srivatsadavuluriiii/FSO-beam-recovery}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

<div align="center">


[⬆ Back to Top](#deep-learning-for-oam-beam-recovery-in-atmospheric-turbulence)

</div>
