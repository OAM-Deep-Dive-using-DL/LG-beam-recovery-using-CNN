# FSO-OAM End-to-End Pipeline Documentation

This document details the step-by-step execution flow of the **LDPC + Pilot + MMSE** FSO-OAM simulation pipeline. It maps the conceptual steps to the specific code files and functions used.

**Root Directory:** `models/LDPC + Pilot + MMSE trials/`

---

## 1. Initialization Phase
**File:** `pipeline.py` (Main Entry Point)

The simulation starts by initializing the core components based on `SimulationConfig`.

1.  **Transmitter Setup** (`encoding.py` -> `encodingRunner`)
    *   **LDPC:** Initializes `PyLDPCWrapper` (Rate 0.8, n=2048).
    *   **Modulation:** Initializes `QPSKModulator`.
    *   **Pilots:** Initializes `PilotHandler` (Comb pattern, 10% ratio).
    *   **Beams:** Pre-computes `LaguerreGaussianBeam` objects for each spatial mode.

2.  **Turbulence Model** (`turbulence.py` -> `AtmosphericTurbulence`)
    *   Calculates Fried parameter ($r_0$) and Rytov variance based on $C_n^2$.
    *   **Phase Screens:** Generates $N$ phase screens using `create_multi_layer_screens` (Split-step Fourier method).

3.  **Simulation Grid**
    *   Defines the spatial grid ($x, y$) and frequency grid ($k_x, k_y$).
    *   **Basis Generation:** Generates the ideal LG beam fields at the transmitter ($z=0$) for all modes.
    *   **Scaling:** Scales basis fields so that total power equals `P_TX_TOTAL_W`.

4.  **Receiver Setup** (`receiver.py` -> `FSORx`)
    *   Initializes `OAMDemultiplexer` (for mode separation).
    *   Initializes `ChannelEstimator` (for pilot-based estimation).
    *   Shares the **same LDPC instance** as the transmitter (critical for correct decoding).

---

## 2. Transmission Phase
**File:** `encoding.py` -> `encodingRunner.transmit()`

Data flows from bits to optical fields:

1.  **Bit Generation:** Random information bits are generated.
2.  **LDPC Encoding:**
    *   `ldpc.encode()`: Adds parity bits to the info bits.
3.  **QPSK Modulation:**
    *   `qpsk.modulate()`: Maps bits to complex QPSK symbols.
4.  **Pilot Insertion:**
    *   `pilot_handler.insert_pilots_per_mode()`: Inserts known pilot symbols (comb pattern) into the data stream for each mode.
5.  **Phase Noise (Optional):**
    *   Adds random phase noise to simulate laser linewidth.
6.  **Frame Creation:**
    *   Packages everything into an `FSO_MDM_Frame` object containing the signals for each mode.

---

## 3. Channel Propagation Phase
**File:** `pipeline.py` -> `run_e2e_simulation()` loop

The physical propagation of the optical beam:

1.  **Multiplexing:**
    *   For each symbol time $t$, the fields of all modes are summed: $E_{tx}(x,y) = \sum a_m \cdot \Psi_m(x,y)$.
2.  **Turbulence Application:**
    *   `turbulence.apply_multi_layer_turbulence()`: Propagates the field through the pre-computed phase screens using Fresnel diffraction (angular spectrum method).
3.  **Attenuation:**
    *   Applies atmospheric loss (Kim model) and geometric loss (beam divergence).
4.  **Noise Addition:**
    *   Adds AWGN (Additive White Gaussian Noise) to the field based on the target SNR.
5.  **Aperture Masking:**
    *   Applies a circular aperture mask (receiver telescope pupil) to the field.

---

## 4. Reception Phase
**File:** `receiver.py` -> `FSORx.receive_frame()`

Recovering bits from the received optical fields:

1.  **OAM Demultiplexing** (`OAMDemultiplexer.project_field`)
    *   **Projection:** Computes the inner product of the received field $E_{rx}$ with the conjugate of the reference mode fields $\Psi_m^*$.
    *   $y_m = \langle E_{rx}, \Psi_m \rangle$
    *   This separates the spatial modes, but they are mixed due to crosstalk (turbulence).

2.  **Channel Estimation** (`ChannelEstimator.estimate_channel_ls`)
    *   Extracts the received pilot symbols $y_p$.
    *   **LS Estimate:** Computes the channel matrix $\hat{H}$ using Least Squares: $\hat{H} = Y_p P_p^H (P_p P_p^H)^{-1}$.

3.  **Noise Estimation**
    *   Estimates noise variance $\sigma^2$ from the metadata (true SNR) or pilot residuals.

4.  **Equalization** (`FSORx` step 5)
    *   **MMSE:** Computes the equalizer matrix $W = \hat{H}^H (\hat{H} \hat{H}^H + \sigma^2 I)^{-1}$.
    *   **Equalize:** $\hat{s} = W \cdot y_{data}$. This mitigates crosstalk.
    *   **Normalization:** Auto-scales output to match unit energy.

5.  **Blind Phase Correction**
    *   **4th Power Method:** Estimates residual phase rotation $\phi$ (piston phase) by averaging $s^4$.
    *   De-rotates symbols: $\hat{s}_{corr} = \hat{s} \cdot e^{-j\phi}$.

6.  **Demodulation** (`QPSKModulator.demodulate_soft`)
    *   Computes **Log-Likelihood Ratios (LLRs)** for each bit based on the distance to QPSK constellation points and noise variance.

7.  **LDPC Decoding** (`PyLDPCWrapper.decode_bp`)
    *   **Belief Propagation:** Iteratively decodes the LLRs to recover the original information bits.

---

## 5. Evaluation Phase
**File:** `pipeline.py`

1.  **BER Calculation:** Compares decoded bits with original transmitted bits.
2.  **Visualization:**
    *   Plots transmitted vs. received constellations.
    *   Visualizes the Channel Matrix ($|H|$ and $\angle H$).
    *   Plots intensity profiles of TX and RX fields.

---

## Key Files Summary

| Component | File | Description |
| :--- | :--- | :--- |
| **Orchestrator** | `pipeline.py` | Runs the loop, manages config, collects results. |
| **Receiver** | `receiver.py` | Demux, Channel Est, MMSE, Decoding logic. |
| **Transmitter** | `encoding.py` | LDPC, QPSK, Pilots, Beam generation. |
| **Physics** | `turbulence.py` | Phase screens, split-step propagation. |
| **Physics** | `lgBeam.py` | Laguerre-Gaussian beam math. |
