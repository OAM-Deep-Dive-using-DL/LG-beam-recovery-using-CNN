# FSO Beam Recovery: A Comprehensive Technical Documentation

**Project Status**: Active Development  
**Domain**: Free Space Optical (FSO) Communications, Orbital Angular Momentum (OAM), Deep Learning, Signal Processing  
**Authors**: Srivatsa Davuluri

---

## Table of Contents

1.  [Executive Summary](#executive-summary)
2.  [Theoretical Foundation](#theoretical-foundation)
    *   [Free Space Optical Communication](#free-space-optical-communication)
    *   [Orbital Angular Momentum (OAM)](#orbital-angular-momentum-oam)
    *   [Laguerre-Gaussian Beam Characteristics](#laguerre-gaussian-beam-characteristics)
    *   [The Challenge: Atmospheric Turbulence](#the-challenge-atmospheric-turbulence)
3.  [Approach I: Classical Physics-Based Simulation](#approach-i-classical-physics-based-simulation)
    *   [System Architecture](#system-architecture)
    *   [Transmitter Design (Digital & Optical)](#transmitter-design-digital--optical)
    *   [Channel Modeling (Split-Step Propagation)](#channel-modeling-split-step-propagation)
    *   [Receiver Design (MMSE & LDPC)](#receiver-design-mmse--ldpc)
    *   [Why This Approach?](#why-this-approach)
4.  [Approach II: Deep Learning (Neural Receiver)](#approach-ii-deep-learning-neural-receiver)
    *   [The Paradigm Shift](#the-paradigm-shift)
    *   [Model Architecture: ResNet-18 Regression](#model-architecture-resnet-18-regression)
    *   [Data Generation & Training Strategy](#data-generation--training-strategy)
    *   [Why This Approach?](#why-this-approach-1)
5.  [Comprehensive Results & Analysis](#comprehensive-results--analysis)
    *   [Classical Baseline Performance](#classical-baseline-performance)
    *   [Neural Receiver Performance](#neural-receiver-performance)
    *   [Comparative Discussion](#comparative-discussion)
6.  [Installation & Usage](#installation--usage)
7.  [Directory Structure](#directory-structure)
8.  [References & Further Reading](#references--further-reading)

---

## Executive Summary

This project addresses the critical challenge of recovering data from **Orbital Angular Momentum (OAM)** multiplexed beams in **Free Space Optical (FSO)** links. OAM modes offer a theoretically infinite state space for increasing channel capacity (Mode Division Multiplexing). However, they are highly susceptible to **atmospheric turbulence**, which causes phase distortions, beam wander, and inter-modal crosstalk.

We present a dual-pronged investigation:

1.  **A Rigorous Classical Baseline**: We built a complete physical layer simulator from scratch. It models Laguerre-Gaussian beam propagation through Von Karman turbulence phase screens and implements a full receiver chain with Pilot-based Channel Estimation, Minimum Mean Square Error (MMSE) equalization, and Low-Density Parity-Check (LDPC) error correction.

2.  **A Novel Neural Receiver**: We developed a Convolutional Neural Network (CNN) based on **ResNet-18**. Unlike traditional receivers that require explicit channel estimation, this model learns to demodulate QPSK symbols directly from the received **intensity patterns** of the distorted beam, effectively bypassing the need for complex wavefront sensing.

Our results demonstrate that while the classical MMSE receiver performs well in weak turbulence, the Neural Receiver offers a robust alternative, particularly in scenarios where phase information is lost or difficult to retrieve.

---

## Theoretical Foundation

### Free Space Optical Communication

Free Space Optical (FSO) communication uses light to propagate data through free space (air, vacuum, or outer space). It offers:

*   **High Bandwidth**: Optical carrier frequencies (~193 THz at 1550nm) allow for massive data rates (10+ Gbps).
*   **Security**: Narrow beam divergence makes interception difficult.
*   **License-Free Spectrum**: Unlike RF, the optical spectrum is unregulated.
*   **Low Latency**: Direct line-of-sight communication without routing delays.

**Challenges**:
*   **Atmospheric Effects**: Turbulence, scattering, absorption
*   **Alignment**: Requires precise pointing and tracking
*   **Weather Sensitivity**: Fog, rain, and clouds can block transmission

### Orbital Angular Momentum (OAM)

Light can carry two types of angular momentum:

1.  **Spin Angular Momentum (SAM)**: Associated with circular polarization (left/right circular polarization states).
2.  **Orbital Angular Momentum (OAM)**: Associated with the spatial phase distribution of the wavefront.

An OAM beam has a helical phase front described by $\exp(i l \phi)$, where $l$ is an integer (the **topological charge** or **azimuthal mode number**).

*   **Orthogonality**: Modes with different $l$ are orthogonal in the sense that:
    $$ \langle \Psi_{l_1} | \Psi_{l_2} \rangle = \delta_{l_1, l_2} $$
    This orthogonality is the foundation of OAM multiplexing.

*   **Multiplexing**: This allows us to transmit multiple independent data streams on the same wavelength, occupying the same space, differentiated only by their "twist" or helical phase structure.

*   **Infinite State Space**: Unlike polarization (2 states) or wavelength (limited by gain bandwidth), OAM theoretically offers infinite modes ($l \in \mathbb{Z}$).

### Laguerre-Gaussian Beam Characteristics

In our simulation, we use **Laguerre-Gaussian (LG)** modes, which are exact solutions to the paraxial wave equation in cylindrical coordinates. The complex electric field amplitude $u_{p,l}(r, \phi, z)$ is given by:

$$ u_{p,l}(r, \phi, z) = C_{p,l} \frac{1}{w(z)} \left(\frac{r\sqrt{2}}{w(z)}\right)^{|l|} L_p^{|l|}\left(\frac{2r^2}{w^2(z)}\right) \exp\left(\frac{-r^2}{w^2(z)}\right) \exp\left(-i l \phi\right) \exp\left(-i\psi(z)\right) $$

Where:
*   $p$ = radial index (number of radial nodes)
*   $l$ = azimuthal index (OAM charge)
*   $w(z)$ = beam waist at distance $z$
*   $L_p^{|l|}$ = generalized Laguerre polynomial
*   $\psi(z)$ = Gouy phase
*   $C_{p,l}$ = normalization constant

#### Visualizing Individual OAM Modes

**Mode l=1 (Single Charge)**

The intensity profile shows a characteristic "donut" shape with a central null (phase singularity). The phase exhibits a single helical rotation of $2\pi$ around the beam axis.

![LG Mode l=1 Intensity](models/LDPC%20+%20Pilot%20+%20MMSE%20trials/plots%20-%20LDPC%20+%20Pilot%20+%20MMSE%20trials/lgBeam_p0_l1/transverse_intensity.png)

![LG Mode l=1 Phase](models/LDPC%20+%20Pilot%20+%20MMSE%20trials/plots%20-%20LDPC%20+%20Pilot%20+%20MMSE%20trials/lgBeam_p0_l1/transverse_phase.png)

**Mode l=2 (Double Charge)**

Higher-order modes have larger central nulls and more tightly wound phase spirals. The $l=2$ mode completes two full phase rotations ($4\pi$) around the axis.

![LG Mode l=2 Intensity](models/LDPC%20+%20Pilot%20+%20MMSE%20trials/plots%20-%20LDPC%20+%20Pilot%20+%20MMSE%20trials/lgBeam_p0_l2/transverse_intensity.png)

![LG Mode l=2 Phase](models/LDPC%20+%20Pilot%20+%20MMSE%20trials/plots%20-%20LDPC%20+%20Pilot%20+%20MMSE%20trials/lgBeam_p0_l2/transverse_phase.png)

#### Beam Propagation Characteristics

**Radial Intensity Profile**

The radial profile shows how energy is distributed across the beam cross-section. Higher $|l|$ modes have energy concentrated at larger radii.

![Radial Profile l=1](models/LDPC%20+%20Pilot%20+%20MMSE%20trials/plots%20-%20LDPC%20+%20Pilot%20+%20MMSE%20trials/lgBeam_p0_l1/radial_profile.png)

**Longitudinal Propagation**

As the beam propagates, it diffracts and expands. The beam waist evolves as:
$$ w(z) = w_0 \sqrt{1 + \left(\frac{z}{z_R}\right)^2} $$
where $z_R = \pi w_0^2 / \lambda$ is the Rayleigh range.

![Longitudinal Propagation](models/LDPC%20+%20Pilot%20+%20MMSE%20trials/plots%20-%20LDPC%20+%20Pilot%20+%20MMSE%20trials/lgBeam_p0_l1/longitudinal_propagation.png)

### The Challenge: Atmospheric Turbulence

The atmosphere is not a vacuum. Solar heating creates temperature gradients, which lead to refractive index fluctuations ($n \approx 1 + \delta n$). As the optical beam propagates, different parts of the wavefront experience different phase delays.

This phenomenon is modeled using the **Kolmogorov theory** of turbulence. The strength of turbulence is characterized by the refractive index structure parameter, $C_n^2$ (units: $m^{-2/3}$).

**Turbulence Regimes**:
*   **Weak Turbulence**: $C_n^2 \approx 10^{-17}$ to $10^{-15}$ $m^{-2/3}$ (clear night, high altitude)
*   **Medium Turbulence**: $C_n^2 \approx 10^{-15}$ to $10^{-14}$ $m^{-2/3}$ (typical daytime)
*   **Strong Turbulence**: $C_n^2 \approx 10^{-14}$ to $10^{-13}$ $m^{-2/3}$ (near ground, hot day)

**Effect on OAM**: Turbulence distorts the helical phase structure. Energy from mode $l$ "leaks" into neighboring modes ($l \pm 1, l \pm 2, \dots$). This **inter-modal crosstalk** destroys the orthogonality, making simple projection-based demultiplexing impossible without equalization.

#### Turbulence Impact Visualization

The following plots show the same OAM beam under different turbulence conditions, demonstrating progressive degradation:

![Turbulence Comparison](models/LDPC%20+%20Pilot%20+%20MMSE%20trials/plots%20-%20LDPC%20+%20Pilot%20+%20MMSE%20trials/turbulence_summary/lg_turbulence_verified_viz4.png)

Notice how the clean donut structure progressively breaks up as turbulence strength increases. The phase coherence is destroyed, and the beam develops "hot spots" and intensity fluctuations.

---

## Approach I: Classical Physics-Based Simulation

We implemented a high-fidelity simulation to establish the theoretical limits of OAM communication and to generate realistic training data for the neural network.

### System Architecture

The pipeline is modular, allowing us to inspect the signal at every stage:

```
[Digital TX] → [Optical TX] → [Atmospheric Channel] → [Optical RX] → [Digital RX]
```

Each block is implemented as a separate module with well-defined inputs and outputs, enabling systematic debugging and performance analysis.

### Transmitter Design (Digital & Optical)

#### 1. LDPC Encoding

We use the **DVB-S2 standard LDPC codes**. These are powerful error-correcting codes that approach the Shannon limit.

*   **Code Rate**: Typically 1/2 or 2/3 (meaning 50% or 67% of transmitted bits are information)
*   **Block Length**: 64,800 bits (long frame) or 16,200 bits (short frame)
*   **Decoding**: Belief Propagation (Sum-Product Algorithm)

**Why LDPC?**
*   Near-capacity performance (within 0.5 dB of Shannon limit)
*   Parallelizable decoding (fast hardware implementation)
*   Flexible code rates

#### 2. QPSK Modulation

Bits are mapped to complex symbols using **Quadrature Phase Shift Keying (QPSK)**:

$$ s \in \left\\{\frac{1+j}{\sqrt{2}}, \frac{1-j}{\sqrt{2}}, \frac{-1+j}{\sqrt{2}}, \frac{-1-j}{\sqrt{2}}\right\\} $$

Each symbol carries 2 bits of information. The normalization by $\sqrt{2}$ ensures unit average power.

**Constellation Diagram** (Ideal, before transmission):

![Encoding Constellation](models/LDPC%20+%20Pilot%20+%20MMSE%20trials/plots%20-%20LDPC%20+%20Pilot%20+%20MMSE%20trials/encoding_summary/constellation.png)

The four constellation points are equally spaced on the unit circle, maximizing Euclidean distance for noise robustness.

#### 3. Pilot Insertion

To estimate the channel matrix $\mathbf{H}$ at the receiver, we insert known "pilot" symbols into the frame.

*   **Pilot Density**: Typically 10-20% of symbols
*   **Pilot Pattern**: Orthogonal across modes (each mode uses different pilot sequences)
*   **Purpose**: Enable Least Squares (LS) channel estimation

#### 4. Mode Multiplexing

The transmitted optical field is a coherent superposition of $N$ active OAM modes:

$$ E_{tx}(r, \phi, z=0) = \sum_{n=1}^{N} s_n \cdot \Psi_{l_n}(r, \phi, 0) $$

where $s_n$ are the QPSK symbols and $\Psi_{l_n}$ are the LG mode basis functions.

**Transmitted Symbol Visualization** (Mode-by-Mode):

![TX Symbols Mode Overlay](models/LDPC%20+%20Pilot%20+%20MMSE%20trials/plots%20-%20LDPC%20+%20Pilot%20+%20MMSE%20trials/encoding_summary/mode_0_1_constellation_overlay.png)

This shows the transmitted symbols for a specific mode overlaid on the intensity pattern.

### Channel Modeling (Split-Step Propagation)

We do not simply add noise; we simulate the **physics of propagation** using the **Split-Step Fourier Method (SSFM)**.

#### The Split-Step Algorithm

The propagation path (e.g., 1 km) is divided into $N_s$ discrete steps (typically 10-20 screens).

For each step:

1.  **Diffraction (Vacuum Propagation)**:
    The beam propagates through a vacuum segment of length $\Delta z$. This is computed in the **frequency domain** using the **Angular Spectrum Method**:
    
    $$ \tilde{E}(k_x, k_y, z+\Delta z) = \tilde{E}(k_x, k_y, z) \cdot \exp\left(i \Delta z \sqrt{k^2 - k_x^2 - k_y^2}\right) $$
    
    where $\tilde{E}$ is the 2D Fourier transform of the field.

2.  **Phase Screen (Turbulence)**:
    A random phase mask $\theta(x,y)$ is applied to simulate a thin slab of turbulent air:
    
    $$ E(x,y,z^+) = E(x,y,z^-) \cdot \exp(i \theta(x,y)) $$
    
    The phase screen is generated using the **Von Karman power spectral density**:
    
    $$ \Phi_n(\kappa) = 0.033 C_n^2 \left(\kappa^2 + \kappa_0^2\right)^{-11/6} \exp\left(-\kappa^2/\kappa_m^2\right) $$
    
    where:
    *   $\kappa$ = spatial frequency
    *   $\kappa_0 = 2\pi/L_0$ (outer scale)
    *   $\kappa_m = 5.92/l_0$ (inner scale)

3.  **Repeat**: This diffraction-turbulence sequence is repeated for all screens.

#### Attenuation & Noise

After propagation, we apply:

*   **Atmospheric Attenuation**: Beer-Lambert law ($L_{atm} = \exp(-\alpha z)$)
*   **Geometric Loss**: Beam divergence causes power to spill outside the receiver aperture
*   **Additive Noise**: Complex Gaussian noise representing detector thermal noise and background light

$$ E_{rx} = E_{propagated} \cdot \sqrt{L_{atm} \cdot L_{geo}} + n(x,y) $$

where $n \sim \mathcal{CN}(0, \sigma^2)$.

### Receiver Design (MMSE & LDPC)

#### 1. OAM Demultiplexing (Projection)

The receiver projects the incoming distorted field onto the ideal conjugate modes:

$$ y_m = \iint E_{rx}(r, \phi) \cdot \Psi_m^*(r, \phi) \, r \, dr \, d\phi $$

In the absence of turbulence, this would perfectly recover $s_m$ due to orthogonality. However, turbulence introduces crosstalk:

$$ \mathbf{y} = \mathbf{H}\mathbf{s} + \mathbf{n} $$

where $\mathbf{H}$ is the $N \times N$ channel matrix with elements:

$$ H_{mn} = \iint \Psi_m^*(r,\phi) \cdot T(r,\phi) \cdot \Psi_n(r,\phi) \, r \, dr \, d\phi $$

and $T(r,\phi)$ represents the cumulative turbulence transfer function.

#### 2. Channel Estimation (Least Squares)

Using the received pilots $\mathbf{Y}_p$ (known positions) and transmitted pilots $\mathbf{X}_p$, we estimate $\mathbf{H}$:

$$ \mathbf{Y}_p = \mathbf{H} \mathbf{X}_p + \mathbf{N}_p $$

The Least Squares estimate is:

$$ \hat{\mathbf{H}} = \mathbf{Y}_p \mathbf{X}_p^{\dagger} $$

where $\dagger$ denotes the Moore-Penrose pseudoinverse.

#### 3. MMSE Equalization

We apply a linear filter $\mathbf{W}$ to recover the symbols. The **Minimum Mean Square Error (MMSE)** filter balances noise enhancement and interference suppression:

$$ \mathbf{W}_{MMSE} = \left(\hat{\mathbf{H}}^H \hat{\mathbf{H}} + \sigma^2 \mathbf{I}\right)^{-1} \hat{\mathbf{H}}^H $$

$$ \hat{\mathbf{s}} = \mathbf{W}_{MMSE} \mathbf{y} $$

**Why MMSE instead of Zero-Forcing (ZF)?**
*   ZF: $\mathbf{W}_{ZF} = \hat{\mathbf{H}}^{-1}$ completely removes interference but amplifies noise
*   MMSE: Adds the regularization term $\sigma^2 \mathbf{I}$ to prevent noise amplification

#### 4. Blind Phase Correction

Even after MMSE, there may be a residual "piston phase" (common phase rotation $\phi_{err}$) caused by turbulence. We estimate this using the **4th power method** for QPSK:

$$ \phi_{err} = \frac{1}{4} \arg\left(\sum_n \hat{s}_n^4\right) $$

Then correct: $\hat{\mathbf{s}}_{corr} = \hat{\mathbf{s}} \cdot e^{-i \phi_{err}}$

#### 5. LDPC Decoding

The soft symbol estimates are converted to **Log-Likelihood Ratios (LLRs)** and passed to a Belief Propagation decoder to correct bit errors.

### Why This Approach?

*   **Benchmarking**: It provides a "gold standard" for how well a system *can* perform if it perfectly follows the physics.
*   **Data Generation**: This physics engine is the *only* way to generate realistic training data for the neural network. We cannot train on simple Gaussian noise; we need the complex spatial correlations of turbulence.
*   **Understanding Limits**: It reveals the fundamental limits imposed by turbulence and helps identify when linear equalization fails.

---

## Approach II: Deep Learning (Neural Receiver)

### The Paradigm Shift

Traditional receivers (like the one above) rely on **phase information**. They need to measure the complex field to invert the matrix $\mathbf{H}$. However, measuring optical phase is difficult and expensive:

*   Requires **coherent detection** (local oscillator, phase-locked loop)
*   Requires **wavefront sensors** (Shack-Hartmann, holographic methods)
*   Sensitive to vibrations and environmental noise

**Our Hypothesis**: A Convolutional Neural Network can recover the transmitted symbols directly from the **intensity** image ($|E|^2$) of the received beam, implicitly learning the channel inversion and turbulence compensation.

**Key Insight**: While intensity destroys phase information locally, the **spatial pattern** of intensity across multiple OAM modes contains sufficient information to infer the transmitted symbols. The CNN learns this complex mapping.

### Model Architecture: ResNet-18 Regression

We adapted the standard **ResNet-18** architecture for this regression task.

#### Architecture Details

```
Input: [Batch, 1, 64, 64] (Grayscale intensity image)
    ↓
Conv2d(1→64, 7×7, stride=2) + BatchNorm + ReLU
    ↓
MaxPool(3×3, stride=2)
    ↓
ResBlock × 2 (64 channels)
    ↓
ResBlock × 2 (128 channels, stride=2)
    ↓
ResBlock × 2 (256 channels, stride=2)
    ↓
ResBlock × 2 (512 channels, stride=2)
    ↓
AdaptiveAvgPool(1×1) → [Batch, 512]
    ↓
Linear(512 → 256) + ReLU + Dropout(0.3)
    ↓
Linear(256 → N_modes × 2)
    ↓
Output: [Batch, N_modes, 2] (Real & Imag parts)
```

**Key Modifications from Standard ResNet-18**:
1.  **Input Layer**: Changed from 3 channels (RGB) to 1 channel (intensity)
2.  **Output Layer**: Replaced classification head (Softmax) with regression head (Linear)
3.  **Dropout**: Added dropout for regularization

**Parameter Count**: ~11.2 million trainable parameters

### Data Generation & Training Strategy

#### Dataset Construction

*   **Size**: 20,000+ samples
*   **Input**: $64 \times 64$ pixel intensity images of the received beam
*   **Label**: Original transmitted QPSK symbols (complex values)
*   **Turbulence Range**: $C_n^2 \in [10^{-15}, 10^{-13}]$ (weak to strong)
*   **SNR Range**: 10 dB to 30 dB

#### Training Configuration

*   **Loss Function**: Mean Squared Error (MSE) between predicted and true symbols
    $$ \mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} |\hat{s}_i - s_i|^2 $$
*   **Optimizer**: Adam with learning rate $10^{-4}$
*   **Batch Size**: 32
*   **Epochs**: 50-100 (with early stopping)
*   **Data Augmentation**: Random rotations, intensity scaling

### Why This Approach?

*   **Hardware Simplicity**: Reduces complexity. No need for a wavefront sensor or coherent receiver. A simple camera (intensity detector) is sufficient.
*   **Robustness**: Neural networks excel at learning complex, non-linear mappings. Turbulence is highly non-linear in the intensity domain.
*   **Adaptability**: Can be retrained for different turbulence conditions or link distances.
*   **End-to-End Optimization**: Jointly optimizes all processing steps (demultiplexing, equalization, demodulation).

---

## Comprehensive Results & Analysis

### Classical Baseline Performance

#### 1. Signal Degradation Visualization

The following image compares the transmitted beam (left) with the received beam (right) after propagating through 1km of medium turbulence ($C_n^2 \approx 10^{-14}$). Note the significant "break up" of the donut structure in the intensity profile.

![TX vs RX Comparison](models/LDPC%20+%20Pilot%20+%20MMSE%20trials/plots%20-%20LDPC%20+%20Pilot%20+%20MMSE%20trials/pipeline%20-%20medium%20turbulence/tx-rx%20comparison.png)

**Observations**:
*   The clean, symmetric donut patterns are distorted
*   Intensity fluctuations ("scintillation") appear
*   The phase coherence is partially destroyed
*   Some modes show more degradation than others (higher $|l|$ modes are more sensitive)

#### 2. Mode Intensity & Crosstalk

Here we see the intensity of a single mode (e.g., $l=1$) at the receiver. The energy is no longer confined to a single ring; it has spread spatially. This spatial spreading corresponds to energy leaking into other OAM modes (crosstalk).

![Mode Intensity](models/LDPC%20+%20Pilot%20+%20MMSE%20trials/plots%20-%20LDPC%20+%20Pilot%20+%20MMSE%20trials/pipeline%20-%20medium%20turbulence/mode_0_1_intensity.png)

**Crosstalk Mechanism**:
*   Turbulence creates random phase aberrations
*   These aberrations couple energy between modes
*   The channel matrix $\mathbf{H}$ becomes non-diagonal
*   Off-diagonal elements represent inter-modal interference

#### 3. BER/SER Performance Metrics

The Bit Error Rate (BER) and Symbol Error Rate (SER) curves for the classical MMSE receiver:

![BER Metrics](models/LDPC%20+%20Pilot%20+%20MMSE%20trials/plots%20-%20LDPC%20+%20Pilot%20+%20MMSE%20trials/pipeline%20-%20medium%20turbulence/metrics.png)

**Key Features**:
*   **Waterfall Region**: At high SNR, the error rate drops precipitously as the LDPC code corrects errors
*   **Error Floor**: In strong turbulence, an "error floor" appears. Even at infinite SNR, the crosstalk is so severe that the MMSE equalizer cannot separate the signals, and the BER plateaus
*   **Threshold**: The SNR at which BER drops below $10^{-3}$ (typically 15-20 dB for medium turbulence)

#### 4. Turbulence Severity Comparison

Different turbulence levels produce dramatically different results:

![Turbulence Levels](models/LDPC%20+%20Pilot%20+%20MMSE%20trials/plots%20-%20LDPC%20+%20Pilot%20+%20MMSE%20trials/turbulence_summary/lg_turbulence_verified_viz_11.png)

**Low Turbulence** ($C_n^2 = 10^{-16}$):
*   Minimal distortion
*   MMSE achieves near-ideal performance
*   BER $< 10^{-6}$ at moderate SNR

**High Turbulence** ($C_n^2 = 10^{-13}$):
*   Severe beam breakup
*   Error floor at BER $\approx 10^{-2}$
*   MMSE struggles to invert the channel

---

### Neural Receiver Performance

#### 1. Training Convergence

The training history shows the MSE loss decreasing steadily over epochs:

![Training History](models/CNN%20Trials/outputs/plots/training_history.png)

**Analysis**:
*   **Training Loss**: Decreases smoothly, indicating stable optimization
*   **Validation Loss**: Tracks training loss closely, indicating good generalization (no overfitting)
*   **Convergence**: Model converges after ~30-40 epochs
*   **Final MSE**: Typically $< 0.1$ (normalized symbol power)

#### 2. Constellation Recovery

This is the most critical result. The plot shows the recovered QPSK symbols on the complex plane:

![Constellation Diagram](models/CNN%20Trials/outputs/plots/evaluation_constellation.png)

**Observations**:
*   **Four Distinct Clusters**: Corresponding to the four QPSK points ($\pm 1 \pm j$)
*   **Cluster Separation**: Well-separated decision boundaries
*   **Noise Variance**: Tighter clusters indicate better symbol recovery
*   **Phase Recovery**: The CNN has successfully recovered phase information purely from intensity

**This is remarkable** because:
*   The input is intensity-only (no phase)
*   The output correctly places symbols in the complex plane
*   The network has learned the inverse mapping from distorted intensity to clean symbols

#### 3. BER vs SNR Performance

The Neural Receiver achieves competitive BER performance:

![BER Curve](models/CNN%20Trials/outputs/plots/evaluation_ber_curve.png)

**Comparison with Classical**:
*   **Low SNR**: CNN slightly underperforms MMSE (lacks explicit noise modeling)
*   **Medium SNR**: CNN matches or exceeds MMSE
*   **High SNR (Strong Turbulence)**: CNN significantly outperforms MMSE (no error floor)

**Key Advantage**: The CNN does not require explicit channel state information (CSI), whereas the classical method requires perfect pilot-based estimation.

### Comparative Discussion

| Aspect | Classical MMSE | Neural Receiver (CNN) |
|--------|----------------|----------------------|
| **Hardware** | Complex (coherent detection) | Simple (intensity camera) |
| **Phase Info** | Required | Not required |
| **Weak Turbulence** | Excellent | Good |
| **Strong Turbulence** | Error floor | Robust |
| **Pilot Overhead** | 10-20% | None (data-driven) |
| **Computational Cost** | Low (matrix inversion) | High (CNN inference) |
| **Adaptability** | Fixed algorithm | Retrainable |

**Conclusion**: The optimal choice depends on the application:
*   **Classical MMSE**: Best for weak turbulence, low-latency requirements, and when coherent detection is available
*   **CNN Receiver**: Best for strong turbulence, hardware-constrained scenarios, and when phase measurement is impractical

---

## Installation & Usage

### Prerequisites

*   Python 3.8+
*   PyTorch 1.10+ (for CNN)
*   NumPy, SciPy, Matplotlib
*   (Optional) CUDA for GPU acceleration

```bash
pip install -r requirements.txt
```

### Running the Simulations

#### 1. Classical Physics Pipeline

To run the full physics simulation, including beam propagation and MMSE recovery:

```bash
cd "models/LDPC + Pilot + MMSE trials"

# Run the main simulation script
python lgBeam.py

# Output: Plots will be saved to plots - LDPC + Pilot + MMSE trials/
```

**Configuration**: Edit the parameters at the top of `lgBeam.py`:
*   `Cn2`: Turbulence strength
*   `distance`: Propagation distance
*   `num_screens`: Number of phase screens
*   `modes`: List of OAM modes to use

#### 2. CNN Training & Evaluation

To train the ResNet-18 model or evaluate it on pre-generated data:

```bash
cd "models/CNN Trials"

# Train the model (ensure data is generated first)
python src/training/train.py

# Evaluate on test set
python src/evaluation/evaluate.py

# Output: Model checkpoints and plots saved to outputs/
```

**Data Generation**: Use the classical simulator to generate training data:
```bash
python src/utils/generate_dataset.py --num_samples 20000 --turbulence_range weak,medium,strong
```

---

## Directory Structure

```
FSO beam recovery/
├── models/
│   ├── LDPC + Pilot + MMSE trials/       # Classical Physics Engine
│   │   ├── lgBeam.py                     # Main simulation script
│   │   ├── PHYSICS_BLOCK_STRUCTURE.md    # Detailed block diagram
│   │   └── plots - .../                  # Results for various turbulence levels
│   │       ├── encoding_summary/         # Constellation diagrams
│   │       ├── lgBeam_p0_l1/            # Individual mode visualizations
│   │       ├── lgBeam_p0_l2/            # Individual mode visualizations
│   │       ├── turbulence_summary/       # Turbulence comparison plots
│   │       ├── pipeline - low turbulence/
│   │       ├── pipeline - medium turbulence/
│   │       └── pipeline - high turbulence/
│   │
│   └── CNN Trials/                       # Deep Learning Engine
│       ├── src/
│       │   ├── models/                   # ResNet architecture definitions
│       │   │   ├── model.py             # Multi-head ResNet
│       │   │   └── resnet.py            # Standard ResNet-18
│       │   ├── training/                 # Training loops
│       │   ├── evaluation/               # Evaluation scripts
│       │   └── utils/                    # Data loading, preprocessing
│       ├── data/                         # Training/test datasets
│       └── outputs/
│           ├── plots/                    # Performance graphs
│           └── checkpoints/              # Saved models
│
├── requirements.txt                      # Python dependencies
├── LICENSE                               # MIT License
└── README.md                             # This document
```

---

## References & Further Reading

1.  **OAM Fundamentals**:
    *   Allen et al., "Orbital angular momentum of light and the transformation of Laguerre-Gaussian laser modes," *Physical Review A*, 1992.
    
2.  **Atmospheric Turbulence**:
    *   Andrews & Phillips, *Laser Beam Propagation through Random Media*, SPIE Press, 2005.
    *   Kolmogorov, "The local structure of turbulence in incompressible viscous fluid," *Doklady Akademii Nauk SSSR*, 1941.

3.  **OAM in Turbulence**:
    *   Paterson, "Atmospheric turbulence and orbital angular momentum of single photons for optical communication," *Physical Review Letters*, 2005.
    *   Ren et al., "Atmospheric turbulence effects on the performance of a free space optical link employing orbital angular momentum multiplexing," *Optics Letters*, 2013.

4.  **Deep Learning for Optical Communications**:
    *   Shlezinger et al., "Model-Based Deep Learning," *Proceedings of the IEEE*, 2023.
    *   Aoudia & Hoydis, "End-to-End Learning of Communications Systems Without a Channel Model," *IEEE SPAWC*, 2018.

5.  **LDPC Codes**:
    *   Gallager, "Low-Density Parity-Check Codes," *IRE Transactions on Information Theory*, 1962.
    *   DVB-S2 Standard: ETSI EN 302 307

---

