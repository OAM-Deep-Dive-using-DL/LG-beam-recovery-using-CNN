
import os
import sys
import numpy as np
import h5py
import torch
from tqdm import tqdm
from scipy.ndimage import zoom

# Add physics module to path
current_dir = os.path.dirname(os.path.abspath(__file__))
# Project root is ../../..
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
physics_dir = os.path.join(project_root, "physics")
sys.path.insert(0, physics_dir)

try:
    from pipeline import run_e2e_simulation, SimulationConfig
except ImportError as e:
    print(f"Error importing pipeline: {e}")
    sys.exit(1)

class DatasetGenerator:
    def __init__(self, output_dir, dataset_name, n_samples, img_size=64):
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        self.n_samples = n_samples
        self.img_size = img_size
        
        os.makedirs(output_dir, exist_ok=True)
        self.h5_path = os.path.join(output_dir, f"{dataset_name}.h5")
        
    def resize_image(self, img_complex, target_size):
        """
        Resize complex field intensity to target size x target_size.
        """
        # Get intensity
        intensity = np.abs(img_complex)**2
        
        # Current size
        h, w = intensity.shape
        scale_h = target_size / h
        scale_w = target_size / w
        
        # Resize using spline interpolation (order 1 = linear is fast/safe)
        intensity_resized = zoom(intensity, (scale_h, scale_w), order=1)
        
        # Normalize (global min/max per image? or fixed?)
        # Standard: Normalize peak to 1.0? 
        # Or preserve relative power?
        # CNNs usually like [0,1].
        if np.max(intensity_resized) > 0:
            intensity_resized /= np.max(intensity_resized)
            
        return intensity_resized.astype(np.float32)

    def generate(self):
        print(f"Generating {self.n_samples} samples into {self.h5_path}...")
        
        # Config Override
        class GenConfig(SimulationConfig):
            N_GRID = 256  # Speed up
            OVERSAMPLING = 2
            
            # We need ~1 symbol per screen for diversity, but pipeline forces frames.
            # We will generate frames, but only sample a few symbols from each frame 
            # to avoid correlation, OR we generate many screens.
            # Best compromise: Run pipeline, get 500 symbols (correlated channel).
            # Save ALL of them. 
            # Repeat with new turbulence.
            # 
            # Wait, if I want 20,000 samples for 'weak turbulence',
            # and I assume coherence time is >> symbol time.
            # Training on correlated samples is OK provided we have enough independent realizations (screens).
            
            # Reduce info bits to minimum to get ~100 symbols per frame
            # 1 LDPC block of data implies ~2048 bits -> 1024 symbols.
            # We can't shrink below 1 block easily.
            # So we get 1024 symbols per run.
            # We need 20 runs to get 20,000 samples.
            # That's manageable!
            
            LDPC_BLOCKS = 1 # Minimum blocks
            # Bits will be auto-calculated by pipeline fix: k * 1
            
            # Turbulence Range
            # We will randomize CN2 in the loop below, so this default doesn't matter much.
            CN2 = 1e-16 
            
            # Disable plotting to save time
            PLOT_DIR = os.path.join(self.output_dir, "debug_plots")
            
        
        # Pre-allocate H5 dataset
        # We don't know exact total yet (depends on yield per run), so we append.
        # Ideally, we create resizable datasets.
        
        mode = 'w'
        
        with h5py.File(self.h5_path, mode) as f:
            # Create expandable datasets
            ds_intensity = f.create_dataset('intensity', shape=(0, self.img_size, self.img_size), 
                                          maxshape=(None, self.img_size, self.img_size), dtype=np.float32)
            ds_symbols = f.create_dataset('symbols', shape=(0, 8, 2), 
                                        maxshape=(None, 8, 2), dtype=np.float32)
            ds_cn2 = f.create_dataset('cn2', shape=(0,), maxshape=(None,), dtype=np.float32)
            
            f.attrs['n_modes'] = 8
            
            # SYSTEMATIC SWEEP: 50 points from 1e-18 to 1e-12
            n_sweep_points = 50
            log_cn2_values = np.linspace(np.log10(1e-18), np.log10(1e-12), n_sweep_points)
            samples_per_point = 400 # 2 frames approx
            
            total_collected = 0
            target_total = n_sweep_points * samples_per_point
            
            pbar = tqdm(total=target_total, unit="samp")
            
            for point_idx, log_cn2 in enumerate(log_cn2_values):
                current_cn2 = 10**log_cn2
                GenConfig.CN2 = current_cn2
                
                # Collect enough samples for this point
                point_collected = 0
                while point_collected < samples_per_point:
                    results = run_e2e_simulation(GenConfig, verbose=False)
                
                    if results is None:
                        continue
                        
                    rx_sequence = results['E_rx_sequence']
                    tx_signals = results['tx_signals']
                    
                    n_syms = len(rx_sequence)
                    modes = sorted(tx_signals.keys())
                    
                    batch_intensity = np.zeros((n_syms, self.img_size, self.img_size), dtype=np.float32)
                    batch_symbols = np.zeros((n_syms, 8, 2), dtype=np.float32)
                    batch_cn2 = np.full((n_syms,), current_cn2, dtype=np.float32)
                    
                    for t in range(n_syms):
                        img_complex = rx_sequence[t]
                        batch_intensity[t] = self.resize_image(img_complex, self.img_size)
                        for m_idx, mode in enumerate(modes):
                            sym = tx_signals[mode]['symbols'][t]
                            batch_symbols[t, m_idx, 0] = sym.real
                            batch_symbols[t, m_idx, 1] = sym.imag
                    
                    ds_intensity.resize(total_collected + n_syms, axis=0)
                    ds_symbols.resize(total_collected + n_syms, axis=0)
                    ds_cn2.resize(total_collected + n_syms, axis=0)
                    
                    ds_intensity[total_collected:] = batch_intensity
                    ds_symbols[total_collected:] = batch_symbols
                    ds_cn2[total_collected:] = batch_cn2
                    
                    total_collected += n_syms
                    point_collected += n_syms
                    pbar.update(n_syms)
                
            pbar.close()
            print(f"saved successfully to {self.h5_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--name", type=str, default="debug_dataset")
    args = parser.parse_args()
    
    out_dir = os.path.join(project_root, "data") # models/CNN Trials/data
    gen = DatasetGenerator(out_dir, args.name, args.samples)
    gen.generate()
