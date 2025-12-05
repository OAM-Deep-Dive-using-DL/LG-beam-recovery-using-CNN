# The "Universal Receiver": Brainstorming Robust Single-Model Architectures

## The Challenge: "Speckle to Symbol"
At strong turbulence (Cn² > 1e-15), the beam's phase structure is destroyed, and the intensity pattern becomes a **speckle pattern**.
- **Weak Turbulence:** Pattern recognition (identifying shapes).
- **Strong Turbulence:** Texture analysis (statistical mapping of speckle to modes).

**Goal:** A single model that transitions seamlessly from shape recognition to texture analysis.

---

## Candidate 1: Swin Transformer V2 (The "Heavy Hitter") 🏆
**Why:** It's a hierarchical Vision Transformer.
- **Local Windows:** Captures fine details (interference fringes) in weak turbulence.
- **Shifted Windows:** Allows information to flow globally, building a full picture of the beam.
- **V2 Improvements:** Specifically designed for high-resolution and stability, handling the high dynamic range of intensity images better.

**Pros:**
- **Global Context:** Can correlate a speckle in the top-left with one in the bottom-right (critical for strong turbulence).
- **SOTA Performance:** Consistently beats ResNets on ImageNet/COCO.
- **Drop-in:** Available in `torchvision.models`.

**Cons:**
- Slower training than ResNet.

---

## Candidate 2: ConvNeXt V2 (The "Modern CNN") 🚀
**Why:** A pure Convolutional Network modernized with Transformer design principles.
- **Large Kernels (7x7):** larger receptive field than ResNet (3x3), capturing more "global" context per layer.
- **Layer Norm & GELU:** More stable training.
- **MAE Pre-training:** ConvNeXt V2 is designed to work well with Masked Autoencoder pre-training, which forces the model to learn robust features from partial info (perfect for turbulence!).

**Pros:**
- **Faster Inference:** Pure ConvNet, very optimized.
- **Robustness:** Known to be more robust to domain shifts (like changing Cn²) than standard ResNets.
- **Simplicity:** No complex attention mechanisms.

---

## Candidate 3: MaxViT (The "Hybrid Beast") 🦍
**Why:** Combines the best of CNNs and Transformers.
- **MBConv Blocks:** Efficient local feature extraction (CNN style).
- **Block Attention:** Global attention on a coarse grid (Transformer style).
- **Grid Attention:** Global attention on a sparse grid.

**Pros:**
- **Multi-Scale:** Explicitly sees the beam at multiple scales simultaneously.
- **Linear Complexity:** Global attention without the quadratic cost of standard ViT.

---

## Candidate 4: EfficientNetV2 (The "Scalable Choice") ⚖️
**Why:** Optimized for training speed and parameter efficiency.
- **Fused-MBConv:** Faster than standard depthwise convs.
- **Progressive Learning:** Designed to handle varying image sizes/regularization during training.

**Pros:**
- **Fast Training:** Iterates quickly.
- **High Capacity:** Large versions (L/XL) have huge capacity to memorize speckle patterns.

---

## Recommendation: The "Swin Unet" Approach

If I had to pick **ONE** method to rule them all, I would propose a **Swin Transformer** backbone.

### Why Swin?
The transition from weak to strong turbulence is a transition from **geometry** to **statistics**.
- **CNNs** (ResNet) are biased towards **geometry** (local shapes).
- **Transformers** (Swin) are unbiased and can learn **long-range statistical dependencies** (which is what speckle analysis requires).

### Proposed Experiment
Replace `ResNet18` with `Swin-T` (Tiny) or `Swin-S` (Small).
- **Input:** 128x128 Intensity
- **Backbone:** Swin Transformer
- **Head:** Linear -> 16 outputs (Real/Imag for 8 modes)

**Hypothesis:** The Self-Attention mechanism will maintain performance in strong turbulence by learning to "attend" to the statistical properties of the speckle field, rather than looking for specific spiral shapes that no longer exist.
