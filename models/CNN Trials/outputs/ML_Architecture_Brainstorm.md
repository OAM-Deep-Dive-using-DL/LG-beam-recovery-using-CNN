# Deep Learning Alternatives for FSO-OAM: Rethinking the Architecture

## Current Limitations & Overhead

### Our Current Approach
```
Intensity Image → CNN → QPSK Symbols → Hard Decisions → Bits
                   ↑
              (with 20% pilot overhead)
```

**Overhead:**
- **20% Pilot Power** → Reduces data throughput by 20%
- **ResNet-18 Bottleneck** → 512-dim may be insufficient for 8 modes × 2 (I/Q)
- **Sequential Processing** → Demux, equalization, demod are separate
- **Intensity-Only Constraint** → No phase diversity

---

## Alternative 1: End-to-End Autoencoder (No Pilots!)

### Concept
Replace the entire TX-RX chain with a learned autoencoder:

```
TX: Bits → Encoder NN → Spatial Field (learned modulation)
RX: Intensity → Decoder NN → Bits (direct bit recovery)
```

### Architecture
```python
# Transmitter
class TxEncoder(nn.Module):
    def __init__(self, n_bits_per_frame=1024, n_modes=8):
        self.bit_to_latent = nn.Sequential(
            nn.Linear(n_bits_per_frame, 512),
            nn.ReLU(),
            nn.Linear(512, n_modes * 2)  # Real/Imag per mode
        )
        # Convert to spatial field using physics (LG basis)
        # Output: [N, N] complex field
        
# Receiver  
class RxDecoder(nn.Module):
    def __init__(self, n_bits_per_frame=1024):
        # Vision Transformer for global context
        self.vit = VisionTransformer(
            img_size=128,
            patch_size=16,  # 8×8 = 64 patches
            in_channels=1,
            embed_dim=512,
            depth=6,
            num_heads=8
        )
        self.bits_head = nn.Linear(512, n_bits_per_frame)
```

### Advantages ✅
1. **No Pilot Overhead** → 100% data throughput
2. **Joint Optimization** → TX and RX co-designed
3. **Learned Modulation** → Not restricted to QPSK (could discover better constellations)
4. **Implicit Channel Estimation** → No separate demux/equalization

### Challenges ⚠️
1. **Training Complexity** → Need to backprop through physics (turbulence simulation)
2. **Hardware Constraint** → TX modulation must be implementable (SLM constraints)
3. **Generalization** → Trained on specific turbulence distribution

### Feasibility: **Medium** (requires differentiable turbulence simulator)

---

## Alternative 2: Vision Transformer with Positional Encoding (Pilot-Free)

### Concept
Use self-attention to capture global interference patterns without explicit pilots:

```
Intensity [128×128] → Patch Embedding [64 patches × 512-dim] 
                    → Transformer Encoder (12 layers)
                    → Multi-Head Output [8 modes × 2 I/Q]
```

### Why Transformers > CNNs for OAM?
1. **Global Receptive Field** → Captures long-range spiral phase dependencies
2. **Permutation Invariance** → Less sensitive to spatial shifts (turbulence)
3. **Attention Mechanism** → Can learn to "focus" on informative regions

### Architecture
```python
class OAMTransformer(nn.Module):
    def __init__(self, n_modes=8, img_size=128, patch_size=16):
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,  # 16×16 patches → 64 patches
            in_channels=1,
            embed_dim=768
        )
        
        # Learnable positional encoding (spiral-aware)
        self.pos_encoding = LearnablePolarPositionalEncoding(
            num_patches=64, embed_dim=768
        )
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=768,
                nhead=12,
                dim_feedforward=3072,
                dropout=0.1,
                activation='gelu'
            ),
            num_layers=12
        )
        
        # Dual heads
        self.symbol_head = nn.Linear(768, n_modes * 2)
        self.confidence_head = nn.Linear(768, n_modes)  # Per-mode SNR estimate
```

### Key Innovation: Polar Positional Encoding
Instead of standard 2D positional encoding, use **(r, θ)** coordinates:
```python
class LearnablePolarPositionalEncoding(nn.Module):
    def forward(self, x, patches_per_side=8):
        # Convert patch indices to (r, θ)
        i, j = patch_coords  # [0-7, 0-7]
        r = sqrt((i - 3.5)^2 + (j - 3.5)^2)
        θ = atan2(j - 3.5, i - 3.5)
        
        # Learnable radial and angular embeddings
        r_embed = self.radial_mlp(r)
        θ_embed = self.angular_mlp(θ)
        return x + r_embed + θ_embed
```

### Advantages ✅
1. **Captures Spiral Structure** → Polar encoding aligns with OAM physics
2. **Global Context** → Sees entire beam simultaneously
3. **Scalable** → Easy to add more modes (just increase output dim)
4. **Interpretable** → Attention maps show which patches matter

### Challenges ⚠️
1. **Computational Cost** → Transformers are 3-5× slower than CNNs
2. **Data Hungry** → Needs 100k+ samples (we have 25k)
3. **Still Needs Phase Info** → Pilot problem remains

### Feasibility: **High** (proven in vision tasks)

---

## Alternative 3: Reinforcement Learning for Adaptive Receiver

### Concept
Frame the receiver as an **agent** that learns to adapt to turbulence:

```
State: Intensity image [128×128]
Action: Equalizer weights [8×8], demod thresholds
Reward: Negative BER (from known training bits)
```

### RL Framework: Soft Actor-Critic (SAC)
```python
class AdaptiveReceiver(nn.Module):
    def __init__(self):
        # Feature extractor (shared)
        self.cnn = ResNet18(output_dim=512)
        
        # Policy network (actor)
        self.actor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 8*8*2)  # Complex equalizer matrix
        )
        
        # Value network (critic)
        self.critic = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Expected reward
        )
```

### Training Loop
```python
for episode in range(num_episodes):
    state = intensity_image
    
    # Agent selects equalizer weights
    action = agent.get_action(state)
    W_eq = action.reshape(8, 8) + 1j*action.reshape(8, 8)
    
    # Apply equalization
    symbols_est = W_eq @ rx_symbols
    bits_est = demodulate(symbols_est)
    
    # Compute reward (negative BER)
    reward = -BER(bits_est, true_bits)
    
    # Update policy
    agent.update(state, action, reward, next_state)
```

### Advantages ✅
1. **Online Adaptation** → Updates weights in real-time during transmission
2. **No Retraining** → Adapts to new turbulence conditions instantly
3. **Interpretable Actions** → Equalizer weights have physical meaning
4. **Meta-Learning** → Can learn to learn (MAML)

### Challenges ⚠️
1. **Reward Engineering** → BER is noisy and delayed
2. **Sample Efficiency** → RL needs many trials
3. **Stability** → Policy can diverge

### Feasibility: **Medium** (interesting research direction)

---

## Alternative 4: Hybrid Physics-Informed Neural Network (PINN)

### Concept
Combine classical signal processing with learned components:

```
Intensity → [Physics: Demux] → Rx Symbols 
                              → [NN: Smart Equalization] → Symbols
                              → [Physics: QPSK Demod] → Bits
```

### Architecture: Neural Equalizer
```python
class PhysicsInformedEqualizer(nn.Module):
    def __init__(self, n_modes=8):
        # Learn a correction to MMSE
        self.mmse_correction = nn.Sequential(
            nn.Linear(n_modes*2, 256),  # F eatures: Re/Im of rx symbols
            nn.ReLU(),
            nn.Linear(256, n_modes*n_modes*2)  # Complex correction matrix
        )
        
    def forward(self, rx_symbols, H_est, noise_var):
        # Classical MMSE
        W_mmse = MMSE_weights(H_est, noise_var)
        
        # Learn residual correction
        features = torch.cat([rx_symbols.real, rx_symbols.imag], dim=-1)
        correction = self.mmse_correction(features)
        W_correction = correction.view(8, 8) + 1j * correction.view(8, 8)
        
        # Combined equalizer
        W = W_mmse + 0.1 * W_correction  # Small correction
        return W @ rx_symbols
```

### Physics Loss Function
```python
def physics_informed_loss(y_pred, y_true, H_est):
    # Standard MSE
    mse = F.mse_loss(y_pred, y_true)
    
    # Physics constraint: ||H @ s_est - y||^2 should be small
    residual = H_est @ y_pred - rx_symbols
    physics_penalty = torch.mean(torch.abs(residual)**2)
    
    # QPSK constellation constraint: symbols should be ±0.707 ± 0.707j
    qpsk_points = torch.tensor([0.707+0.707j, -0.707+0.707j, ...])
    constellation_loss = min_distance_to_qpsk(y_pred, qpsk_points)
    
    return mse + 0.1*physics_penalty + 0.05*constellation_loss
```

### Advantages ✅
1. **Best of Both Worlds** → Classical foundation + learned refinement
2. **Interpretable** → Can inspect physics vs learned contributions
3. **Data Efficient** → Physics prior reduces training needs
4. **Stable** → Classical fallback if NN fails

### Challenges ⚠️
1. **Complexity** → Need to carefully balance physics vs learning
2. **Tuning** → Loss weights (0.1, 0.05) are hyperparameters

### Feasibility: **High** (proven in other domains like fluid dynamics)

---

## Alternative 5: Generative Model for Turbulence Removal

### Concept
Instead of recovering symbols, **denoise the intensity image first**:

```
Turbulent Intensity → [Diffusion Model / VAE] → Clean Intensity
                                                → [Classical Receiver] → Bits
```

### Architecture: Denoising Diffusion Probabilistic Model (DDPM)
```python
class TurbulenceDDPM(nn.Module):
    def __init__(self):
        self.unet = UNet(
            in_channels=1,
            out_channels=1,
            channels=[64, 128, 256, 512],
            attention_levels=[2, 3]  # Attend at 32×32 and 16×16
        )
        
    def forward(self, noisy_image, timestep):
        # Predict noise added by turbulence
        noise_pred = self.unet(noisy_image, timestep)
        return noisy_image - noise_pred  # Denoised
```

### Training
```python
# Generate pairs: (clean, turbulent) images
for clean_intensity, turbulent_intensity in dataset:
    # Add timestep noise (simulating diffusion process)
    t = random.randint(0, T)
    noise = torch.randn_like(clean_intensity)
    noisy = sqrt(alpha_t) * clean_intensity + sqrt(1-alpha_t) * noise
    
    # Train to predict noise
    noise_pred = model(noisy, t)
    loss = F.mse_loss(noise_pred, noise)
```

### Advantages ✅
1. **Modular** → Denoising is separate from symbol recovery
2. **Reusable** → Can use classical receiver on cleaned images
3. **Interpretable** → Can visualize denoised images
4. **SOTA in Image Restoration** → Proven in medical imaging, astronomy

### Challenges ⚠️
1. **Computationally Expensive** → DDPM needs 50-1000 denoising steps
2. **Latency** → Too slow for real-time (unless using fast samplers)
3. **Overfitting Risk** → Turbulence patterns may be too complex

### Feasibility: **Medium-High** (worth exploring)

---

## Alternative 6: Meta-Learning for Fast Adaptation

### Concept
Train a model that can **quickly adapt to new turbulence** with just a few samples:

```
Meta-Train: Learn initialization θ* that adapts fast
Meta-Test: Fine-tune on 10 samples from new Cn² → θ_new
```

### Algorithm: Model-Agnostic Meta-Learning (MAML)
```python
# Meta-training loop
for task in turbulence_tasks:  # Each task = different Cn²
    # Sample support set (10 samples) and query set (100 samples)
    support_X, support_y = task.sample_support(k=10)
    query_X, query_y = task.sample_query(n=100)
    
    # Inner loop: Fast adaptation
    θ_adapted = θ - α * grad(loss(support_X, support_y; θ), θ)
    
    # Outer loop: Meta-update
    meta_loss = loss(query_X, query_y; θ_adapted)
    θ = θ - β * grad(meta_loss, θ)
```

### Advantages ✅
1. **Rapid Adaptation** → Can handle new turbulence with 10 samples
2. **Few-Shot Learning** → No need for massive dataset per condition
3. **Deployable** → Update model on-the-fly during transmission

### Challenges ⚠️
1. **Meta-Training Complexity** → Needs diverse turbulence tasks
2. **Computational Cost** → Double gradient (hessian)

### Feasibility: **Medium** (active research area)

---

## Comparison Matrix

| Approach | Pilot Overhead | Training Complexity | Inference Speed | Strong Turb | Feasibility |
|:---------|:---------------|:-------------------|:----------------|:------------|:------------|
| **Current (ResNet)** | 20% | Low | Fast | ❌ Poor | ✅ Done |
| **Autoencoder** | 0% | Very High | Fast | ❓ Unknown | ⚠️ Medium |
| **Transformer** | 20%* | Medium | Slow | ❓ Unknown | ✅ High |
| **RL Adaptive** | Variable | High | Fast | ✅ Good | ⚠️ Medium |
| **PINN Hybrid** | 20% | Medium | Fast | ✅ Good | ✅ High |
| **Diffusion Denoiser** | 0% | High | Very Slow | ✅ Excellent | ⚠️ Medium |
| **Meta-Learning** | 20% | Very High | Fast | ✅ Good | ⚠️ Medium |

*Transformer could potentially work without pilots using self-supervised learning

---

## Recommended Next Steps

### Short-Term (1-2 weeks)
1. **Try Vision Transformer** with polar positional encoding
   - Drop-in replacement for ResNet-18
   - Use same training pipeline
   - Expected: +10-20% improvement in weak turbulence

2. **Implement PINN Hybrid**
   - Keep classical demux/channel estimation
   - Add neural equalizer with physics loss
   - Expected: +20-30% improvement in moderate turbulence

### Medium-Term (1-2 months)
3. **Pilot-Free Autoencoder**
   - Requires differentiable turbulence simulator
   - Start with simulated data only
   - Expected: Eliminate 20% pilot overhead

4. **Meta-Learning for Adaptation**
   - Train on 15 Cn² values
   - Test fast adaptation to new Cn²
   - Expected: Robust generalization

### Long-Term (3-6 months)
5. **Diffusion Model for Denoising**
   - Pre-train on large astronomy dataset (similar physics)
   - Fine-tune on FSO-OAM
   - Expected: SOTA strong turbulence performance

---

## The Fundamental Question: Do We Need Pilots at All?

### Self-Supervised Learning Approach
```python
# Insight: Adjacent frames have similar turbulence
# Use frame t-1 to predict frame t

class SelfSupervisedOAM(nn.Module):
    def forward(self, frame_t_minus_1, frame_t):
        # Encode previous frame
        context = self.encoder(frame_t_minus_1)
        
        # Decode current frame
        symbols_t = self.decoder(frame_t, context)
        
        # Loss: Consistency with demodulated bits
        # (requires bits to be constant across frames)
        return symbols_t
```

**Key Insight:** If we transmit the same  data across multiple frames (or slowly changing data), we can use **temporal consistency** as a supervisory signal, eliminating pilots entirely!

---

## Conclusion

**Our current approach (ResNet + 20% pilots) is a solid baseline, but has clear limitations.**

**Most Promising Alternatives:**
1. **Vision Transformer** (easy upgrade, likely works)
2. **PINN Hybrid** (best of classical + learning)
3. **Pilot-Free Autoencoder** (highest potential, highest risk)

**The "No Pilot" approaches are especially attractive** because they directly address your concern about overhead. The key is finding a supervisory signal that doesn't require explicit pilots—either through:
- Self-supervised learning (temporal consistency)
- Physics constraints (channel matrix properties)
- Generative modeling (learning turbulence distribution)

**Recommendation:** Start with Transformer + PINN Hybrid (both are feasible), then explore pilot-free methods if those succeed.
