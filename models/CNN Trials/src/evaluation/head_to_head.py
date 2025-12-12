import torch
import numpy as np
import sys
import os
from pathlib import Path

# Add paths
sys.path.append(os.path.abspath("models/CNN Trials/src"))
sys.path.append(os.path.abspath("models/CNN Trials/src/models"))

# Import Physics
sys.path.append(os.path.abspath("models/CNN Trials/physics"))
# Also need project root in path for pipeline to find receiver? No, pipeline is in physics.
from pipeline import run_e2e_simulation, SimulationConfig

# Create a config similar to GenConfig
class LiveConfig(SimulationConfig):
    N_GRID = 256
    OVERSAMPLING = 2
    LDPC_BLOCKS = 1
    PLOT_DIR = "debug_plots_headhead"
    CN2 = 1e-16 # Default

# Import Model
from model import MultiHeadResNet

# Configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load Pre-trained CNN
model = MultiHeadResNet(n_modes=8).to(device)
try:
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    print("Loaded CNN model.")
except:
    print("Error: best_model.pth not found!")
    sys.exit(1)
model.eval()

def qpsk_ber(preds, targets):
    # preds, targets: complex arrays [N]
    # Hard decision
    t_re = np.sign(np.real(targets))
    t_im = np.sign(np.imag(targets))
    p_re = np.sign(np.real(preds))
    p_im = np.sign(np.imag(preds))
    errors = (t_re != p_re).astype(int) + (t_im != p_im).astype(int)
    return np.sum(errors) / (len(targets) * 2)

def eval_point(cn2_val, n_frames=2):
    LiveConfig.CN2 = cn2_val
    
    total_bits = 0
    cnn_errors = 0
    mmse_errors = 0
    
    # We need to capture MMSE results from the pipeline return value
    # But run_e2e_simulation returns a dict with 'results'.
    # IMPORTANT: The current run_e2e_simulation calculates MMSE BER and prints it, 
    # but we need to fetch the raw Recovered Bits or calculated metrics.
    
    # Let's peek at pipeline.py's return:
    # results = { ..., 'ber': metrics['ber'], 'recovered_bits': recovered_bits, ... }
    # So we can just read 'ber' from the result for MMSE!
    
    # Wait, run_e2e_simulation runs ONE simulation frame.
    # The dictionary key is 'ber', which is the MMSE BER.
    
    mmse_ber_accum = 0.0
    
    for _ in range(n_frames):
        # 1. Run Physics (Classical MMSE happens here!)
        sim_res = run_e2e_simulation(LiveConfig, verbose=False)
        if sim_res is None: continue
        
        # Get MMSE Performance (Baseline)
        # Note: sim_res['ber'] is the classical MMSE BER calculated inside receiver.py
        mmse_ber_raw = sim_res['metrics']['ber'] 
        mmse_ber_accum += mmse_ber_raw
        
        # 2. Run CNN
        rx_sequence = sim_res['E_rx_sequence'] # [205, Nx, Ny] complex
        tx_signals = sim_res['tx_signals']     # dict of symbols
        
        # Prepare Batch
        # Resize to 64x64
        # (Using logic from dataset.py/generate_dataset.py)
        # Simple center crop or zoom. generated_dataset uses scipy.zoom usually.
        # But wait, generate_dataset.py defines 'resize_image'. I'll copy the logic.
        from scipy.ndimage import zoom
        
        n_syms = len(rx_sequence)
        
        # Collect Targets
        batch_targets = np.zeros((n_syms, 8), dtype=complex)
        modes = sorted(tx_signals.keys())
        for m_idx, mode in enumerate(modes):
            batch_targets[:, m_idx] = tx_signals[mode]['symbols']
            
        # Collect Inputs
        batch_imgs = np.zeros((n_syms, 1, 64, 64), dtype=np.float32)
        
        for t in range(n_syms):
            field = rx_sequence[t]
            # Calculate Intensity
            intensity = np.abs(field)**2
            # Resize
            # Original grid is 256x256. Target 64x64. Factor 0.25.
            # Zoom is cleaner
            img_small = zoom(intensity, 0.25, order=1)
            # Normalize
            img_small = img_small / (np.max(img_small) + 1e-8)
            batch_imgs[t, 0] = img_small
            
        # Infer CNN
        t_imgs = torch.from_numpy(batch_imgs).float().to(device)
        with torch.no_grad():
            pred_syms, _ = model(t_imgs) # [N, 8, 2]
            
        pred_complex = pred_syms[..., 0] + 1j * pred_syms[..., 1]
        pred_complex = pred_complex.cpu().numpy()
        
        # Calculate CNN BER
        batch_cnn_errors = 0
        total_bits_batch = n_syms * 8 * 2
        
        for m in range(8):
            batch_cnn_errors += qpsk_ber(pred_complex[:, m], batch_targets[:, m]) * (n_syms * 2)
            
        cnn_errors += batch_cnn_errors
        total_bits += total_bits_batch
        
    avg_mmse_ber = mmse_ber_accum / n_frames
    final_cnn_ber = cnn_errors / total_bits
    
    return avg_mmse_ber, final_cnn_ber

if __name__ == "__main__":
    points = [1e-17, 1e-16, 5e-16, 1e-15, 5e-15, 1e-14]
    
    print(f"\n{'Cn2 (m^-2/3)':<15} | {'MMSE BER':<10} | {'CNN BER':<10} | {'Status'}")
    print("-" * 55)
    
    results = []
    
    for cn2 in points:
        mmse_ber, cnn_ber = eval_point(cn2, n_frames=2) # 2 frames = 410 symbols
        
        status = "TIE"
        if cnn_ber < mmse_ber/2: status = "CNN WIN"
        if mmse_ber < cnn_ber/2: status = "MMSE WIN"
        
        print(f"{cn2:.2e}        | {mmse_ber:.4f}     | {cnn_ber:.4f}     | {status}")
        results.append((cn2, mmse_ber, cnn_ber))
        
    print("\nVerification Complete.")
