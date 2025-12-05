"""
Dataset Generator for CNN-based FSO-OAM Receiver
Optimized for Apple M3 (Multiprocessing + Chunked I/O)

Generates training/validation/test datasets by simulating:
1. LG beam generation at transmitter
2. QPSK symbol modulation
3. Atmospheric turbulence propagation
4. Intensity pattern capture at receiver

Saves as HDF5 with:
- Input: 64×64 intensity images
- Labels: QPSK symbols (8 modes × 2 for I/Q)
- Metadata: Cn², turbulence params, etc.
"""

import os
import sys
import json
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import zoom
import argparse
import multiprocessing as mp
from functools import partial

# Add physics modules to path
SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR / 'physics'))

from lgBeam import LaguerreGaussianBeam
from turbulence import AtmosphericTurbulence, create_multi_layer_screens, apply_multi_layer_turbulence
from fsplAtmAttenuation import calculate_kim_attenuation, calculate_geometric_loss

# Global storage for worker processes
WORKER_CONTEXT = {}

def init_worker(context):
    """Initialize worker process with shared data."""
    global WORKER_CONTEXT
    WORKER_CONTEXT.update(context)
    # Re-seed random number generator for each worker to ensure independence
    # mixing base seed with process ID
    base_seed = context.get('random_seed', 42)
    np.random.seed(base_seed + os.getpid())

def _add_noise(intensity, config):
    """Add Gaussian noise to intensity image."""
    if not config.get('augmentation', {}).get('add_noise', False):
        return intensity
        
    # Simple AWGN model
    # Assume a target SNR for the image
    target_snr_db = 20  # dB
    signal_power = np.mean(intensity**2)
    if signal_power <= 0:
        return intensity
        
    noise_power = signal_power / (10**(target_snr_db/10))
    noise_std = np.sqrt(noise_power)
    
    noise = np.random.normal(0, noise_std, intensity.shape)
    noisy_intensity = intensity + noise
    
    # Clip to non-negative
    return np.maximum(noisy_intensity, 0.0)


def _zoom_to_aperture(intensity, grid_info, receiver_diameter):
    """
    Crop the intensity field to the receiver aperture before downsampling.
    
    This "Smart Zoom" dramatically improves effective resolution by:
    - Removing empty space around the beam
    - Increasing pixels/beam diameter from ~3.4 to ~8.2+
    
    Args:
        intensity: [N, N] intensity field
        grid_info: Dict with 'D', 'delta', 'N', 'x', 'y', 'X', 'Y', 'R', 'PHI'
        receiver_diameter: Receiver aperture diameter (m)
    
    Returns:
        intensity_cropped: [N_crop, N_crop] cropped intensity
    """
    # Find grid indices corresponding to receiver aperture
    # Aperture extends from -D_rx/2 to +D_rx/2
    D_rx = receiver_diameter
    delta = grid_info['delta']
    
    # Calculate how many pixels correspond to the aperture
    # Need to be conservative and round outward
    half_pixels = int(np.ceil(D_rx / (2 * delta)))
    center_idx = grid_info['N'] // 2
    
    # Extract indices
    i_min = max(0, center_idx - half_pixels)
    i_max = min(grid_info['N'], center_idx + half_pixels)
    j_min = max(0, center_idx - half_pixels)
    j_max = min(grid_info['N'], center_idx + half_pixels)
    
    # Crop
    intensity_cropped = intensity[i_min:i_max, j_min:j_max]
    
    return intensity_cropped


def generate_single_sample(cn2_value):
    """
    Worker function to generate a single sample.
    Uses global WORKER_CONTEXT for static data.
    """
    ctx = WORKER_CONTEXT
    sys_params = ctx['sys_params']
    turb_params = ctx['turb_params']
    grid_params = ctx['grid_params']
    data_format = ctx['data_format']
    grid_info = ctx['grid_info']
    basis_fields = ctx['basis_fields']
    lg_beams_keys = ctx['lg_beams_keys']
    max_m2_beam = ctx['max_m2_beam']
    config = ctx['config']
    
    # Generate random QPSK symbols
    qpsk_constellation = np.array([
        (1 + 1j) / np.sqrt(2),   # 00
        (-1 + 1j) / np.sqrt(2),  # 01
        (1 - 1j) / np.sqrt(2),   # 10
        (-1 - 1j) / np.sqrt(2),  # 11
    ])
    
    symbols = np.random.choice(qpsk_constellation, size=len(sys_params['spatial_modes']))
    
    # Create multiplexed TX field
    E_tx = np.zeros((grid_info['N'], grid_info['N']), dtype=complex)
    
    # Calculate power scaling
    total_power = sys_params['p_tx_total']
    pilot_params = sys_params.get('pilot_parameters', {'enabled': False})
    
    if pilot_params['enabled']:
        pilot_ratio = pilot_params.get('power_ratio', 0.1)
        pilot_power = total_power * pilot_ratio
        signal_power = total_power * (1 - pilot_ratio)
        
        # Add Pilot (Mode 0,0 usually)
        pilot_mode_key = tuple(pilot_params.get('mode', [0, 0]))
        # Check if pilot beam is pre-calculated, if not, we might need to generate it or assume it's in basis if listed
        # Ideally, we should generate the pilot beam field here if it's not in basis_fields
        # But for efficiency, let's assume we need to generate it if not present.
        # However, `basis_fields` only contains data modes.
        # Let's generate pilot field on the fly or pre-calc it. 
        # Better: Pre-calc in init, but for now inside worker is safer for independence if not passed.
        # Actually, let's look at `ctx`. We can add pilot beam to `basis_fields` in `_generate_basis_fields`?
        # No, let's just generate it here or pass it.
        # To avoid re-generating every time, let's assume it's passed in `ctx` if we update `_generate_basis_fields`.
        # For now, let's use the `basis_fields` if it happens to be there, or generate it.
        # WAIT: `basis_fields` are scaled for signal power. Pilot has different power.
        # Let's use the unscaled beams from `lg_beams`? No, `lg_beams` are objects.
        # We need the field.
        
        # Let's rely on `basis_fields` containing the pilot mode IF we add it to the list of modes? 
        # No, pilot is separate.
        
        # Let's just generate it. It's fast.
        # We need the beam object. `lg_beams` only has data modes.
        # We need to instantiate the pilot beam.
        # This is inefficient to do per sample.
        # Let's update `_init_beams` and `_generate_basis_fields` instead?
        # Yes, that's cleaner. But I am editing `generate_single_sample` right now.
        
        # Let's assume `ctx` has `pilot_field` if enabled.
        if 'pilot_field' in ctx:
             E_tx += ctx['pilot_field'] * np.sqrt(pilot_power) # pilot_field should be unit power
        
    else:
        signal_power = total_power

    # Add Data Signals
    n_modes = len(sys_params['spatial_modes'])
    power_per_mode = signal_power / n_modes
    scale_per_mode = np.sqrt(power_per_mode)
    
    for i, mode_key in enumerate(lg_beams_keys):
        # basis_fields are currently scaled to (total_power / n_modes). 
        # We need to rescale them if pilot is present or just use unit-power basis.
        # The current `basis_fields` in `_generate_basis_fields` uses `p_tx_total`.
        # We should change `_generate_basis_fields` to return UNIT power fields.
        # Then scale here.
        
        # Assuming `basis_fields` are UNIT POWER (we will update `_generate_basis_fields` next):
        E_tx += basis_fields[mode_key] * symbols[i] * scale_per_mode
    
    # Create turbulence layers
    layers = create_multi_layer_screens(
        total_distance=sys_params['distance'],
        num_screens=turb_params['num_screens'],
        wavelength=sys_params['wavelength'],
        ground_Cn2=cn2_value,
        L0=turb_params['L0'],
        l0=turb_params['l0_inner'],
        cn2_model=turb_params['cn2_model'],
        verbose=False
    )
    
    # Propagate through turbulence
    result = apply_multi_layer_turbulence(
        initial_field=E_tx,
        base_beam=max_m2_beam,
        layers=layers,
        total_distance=sys_params['distance'],
        N=grid_info['N'],
        oversampling=grid_params['oversampling'],
        L0=turb_params['L0'],
        l0=turb_params['l0_inner']
    )
    
    E_rx = result['final_field']
    
    # Apply atmospheric attenuation
    visibility_km = 23.0  # Clear air
    alpha_dBkm = calculate_kim_attenuation(sys_params['wavelength'] * 1e9, visibility_km)
    L_atm_dB = alpha_dBkm * (sys_params['distance'] / 1000.0)
    amplitude_loss = 10**(-L_atm_dB / 20.0)
    
    E_rx = E_rx * amplitude_loss
    
    # Apply receiver aperture
    receiver_radius = sys_params['receiver_diameter'] / 2.0
    aperture_mask = (grid_info['R'] <= receiver_radius).astype(float)
    E_rx = E_rx * aperture_mask
    
    # Compute intensity
    intensity = np.abs(E_rx)**2
    
    # Add noise BEFORE zooming (simulating sensor noise)
    intensity = _add_noise(intensity, config)
    
    # SMART ZOOM: Crop to receiver aperture before downsampling
    # This dramatically improves effective resolution
    intensity_zoomed = _zoom_to_aperture(
        intensity, 
        grid_info, 
        sys_params['receiver_diameter']
    )
    
    # Downsample to output size
    n_out = grid_params['n_grid_output']
    # Now scale factor is relative to the zoomed grid, not the full sim grid
    N_zoomed = intensity_zoomed.shape[0]
    scale_factor = n_out / N_zoomed
    
    
    if grid_params['downsampling_method'] == 'bilinear':
        intensity_downsampled = zoom(intensity_zoomed, scale_factor, order=1)
    else:
        intensity_downsampled = zoom(intensity_zoomed, scale_factor, order=0)
    
    # Normalize if requested
    if data_format['normalize_input']:
        method = data_format['normalization_method']
        if method == 'per_sample':
            min_val = np.min(intensity_downsampled)
            max_val = np.max(intensity_downsampled)
            intensity_downsampled = (intensity_downsampled - min_val) / (max_val - min_val + 1e-10)
    
    # Metadata
    metadata = {
        'cn2': cn2_value,
        'distance': sys_params['distance'],
        'wavelength': sys_params['wavelength'],
        'attenuation_dB': L_atm_dB
    }
    
    return intensity_downsampled, symbols, metadata


class DatasetGenerator:
    """Generate FSO-OAM dataset for CNN training."""
    
    def __init__(self, config_path):
        """
        Initialize dataset generator.
        
        Args:
            config_path: Path to configuration JSON file
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Extract parameters
        self.sys_params = self.config['system_parameters']
        self.turb_params = self.config['turbulence_parameters']
        self.grid_params = self.config['grid_parameters']
        self.data_format = self.config['data_format']
        
        # Set random seed
        np.random.seed(self.config.get('random_seed', 42))
        
        # Initialize LG beams
        self._init_beams()
        
        # Generate Cn² values
        self._generate_cn2_values()
        
        print(f"✓ Dataset generator initialized")
        print(f"  Modes: {len(self.sys_params['spatial_modes'])}")
        print(f"  Cn² range: {self.turb_params['cn2_min']:.2e} to {self.turb_params['cn2_max']:.2e}")
        print(f"  Grid: {self.grid_params['n_grid_sim']}×{self.grid_params['n_grid_sim']} → {self.grid_params['n_grid_output']}×{self.grid_params['n_grid_output']}")
    
    def _init_beams(self):
        """Initialize LG beams for all spatial modes."""
        self.lg_beams = {}
        for mode in self.sys_params['spatial_modes']:
            p, l = mode
            beam = LaguerreGaussianBeam(
                p=p, l=l,
                wavelength=self.sys_params['wavelength'],
                w0=self.sys_params['w0']
            )
            self.lg_beams[tuple(mode)] = beam
        
        # Find beam with largest M² for grid sizing
        self.max_m2_beam = max(self.lg_beams.values(), key=lambda b: b.M_squared)
    
    def _generate_cn2_values(self):
        """Generate Cn² values for dataset."""
        cn2_min = self.turb_params['cn2_min']
        cn2_max = self.turb_params['cn2_max']
        num_points = self.turb_params['num_cn2_points']
        
        # Logarithmically spaced
        self.cn2_values = np.logspace(
            np.log10(cn2_min),
            np.log10(cn2_max),
            num_points
        )
        
        print(f"  Cn² values ({len(self.cn2_values)}): {self.cn2_values[0]:.2e} ... {self.cn2_values[-1]:.2e}")
    
    def _setup_grid(self):
        """Setup simulation grid."""
        distance = self.sys_params['distance']
        # CRITICAL FIX: Use physical beam radius (with M^2) for grid sizing
        # This prevents clipping of higher-order modes
        beam_size_at_rx = self.max_m2_beam.physical_beam_radius(distance)
        
        # Grid extent
        D = self.grid_params['oversampling'] * 6 * beam_size_at_rx
        N = self.grid_params['n_grid_sim']
        delta = D / N
        
        # Create grid
        x = np.linspace(-D/2, D/2, N)
        y = np.linspace(-D/2, D/2, N)
        X, Y = np.meshgrid(x, y, indexing='ij')
        R = np.sqrt(X**2 + Y**2)
        PHI = np.arctan2(Y, X)
        
        return {
            'D': D, 'delta': delta, 'N': N,
            'x': x, 'y': y, 'X': X, 'Y': Y,
            'R': R, 'PHI': PHI
        }
    
    def _generate_basis_fields(self, grid_info):
        """Generate unit-power basis fields for data modes."""
        dA = grid_info['delta']**2
        
        basis_fields = {}
        for mode_key, beam in self.lg_beams.items():
            # Generate field at z=0
            E_basis = beam.generate_beam_field(grid_info['R'], grid_info['PHI'], 0)
            
            # Normalize to unit power
            energy = np.sum(np.abs(E_basis)**2) * dA
            if energy > 0:
                E_basis /= np.sqrt(energy)
            
            basis_fields[mode_key] = E_basis
        
        return basis_fields

    def _generate_pilot_field(self, grid_info):
        """Generate unit-power pilot field."""
        pilot_params = self.sys_params.get('pilot_parameters', {'enabled': False})
        if not pilot_params['enabled']:
            return None
            
        p, l = pilot_params.get('mode', [0, 0])
        beam = LaguerreGaussianBeam(
            p=p, l=l,
            wavelength=self.sys_params['wavelength'],
            w0=self.sys_params['w0']
        )
        
        E_pilot = beam.generate_beam_field(grid_info['R'], grid_info['PHI'], 0)
        
        # Normalize
        dA = grid_info['delta']**2
        energy = np.sum(np.abs(E_pilot)**2) * dA
        if energy > 0:
            E_pilot /= np.sqrt(energy)
            
        return E_pilot
    
    def generate_dataset(self, num_samples, split='train', output_path=None):
        """
        Generate dataset using multiprocessing and chunked I/O.
        """
        if output_path is None:
            dataset_name = self.config['dataset_name']
            output_path = SCRIPT_DIR / 'dataset' / f'{dataset_name}_{split}.h5'
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Generating {split.upper()} dataset: {num_samples} samples")
        print(f"Optimization: Multiprocessing + Chunked I/O")
        print(f"{'='*60}")
        
        # Setup grid (once)
        grid_info = self._setup_grid()
        basis_fields = self._generate_basis_fields(grid_info)
        pilot_field = self._generate_pilot_field(grid_info)
        
        # Prepare worker context
        context = {
            'sys_params': self.sys_params,
            'turb_params': self.turb_params,
            'grid_params': self.grid_params,
            'data_format': self.data_format,
            'grid_info': grid_info,
            'basis_fields': basis_fields,
            'pilot_field': pilot_field,
            'lg_beams_keys': list(self.lg_beams.keys()),
            'max_m2_beam': self.max_m2_beam,
            'config': self.config,
            'random_seed': self.config.get('random_seed', 42)
        }
        
        # Determine Cn² distribution
        samples_per_cn2 = num_samples // len(self.cn2_values)
        
        # Create task list
        tasks = []
        for cn2 in self.cn2_values:
            tasks.extend([cn2] * samples_per_cn2)
        
        # Adjust for rounding errors
        if len(tasks) < num_samples:
            tasks.extend([self.cn2_values[-1]] * (num_samples - len(tasks)))
        tasks = tasks[:num_samples]
        
        # HDF5 Setup
        n_out = self.grid_params['n_grid_output']
        n_modes = len(self.sys_params['spatial_modes'])
        
        chunk_size = 1000  # Write to disk every 1000 samples
        
        with h5py.File(output_path, 'w') as f:
            # Create resizable datasets
            dset_intensity = f.create_dataset('intensity', shape=(0, n_out, n_out), 
                                            maxshape=(None, n_out, n_out),
                                            dtype=np.float32,
                                            chunks=(100, n_out, n_out),
                                            compression='gzip', compression_opts=4)
            
            dset_symbols = f.create_dataset('symbols', shape=(0, n_modes, 2),
                                          maxshape=(None, n_modes, 2),
                                          dtype=np.float32,
                                          chunks=(100, n_modes, 2),
                                          compression='gzip', compression_opts=4)
            
            dset_cn2 = f.create_dataset('cn2', shape=(0,),
                                      maxshape=(None,),
                                      dtype=np.float32)
            
            # Metadata
            f.attrs['split'] = split
            f.attrs['n_modes'] = n_modes
            f.attrs['input_shape'] = [n_out, n_out]
            f.attrs['wavelength'] = self.sys_params['wavelength']
            f.attrs['distance'] = self.sys_params['distance']
            f.attrs['spatial_modes'] = self.sys_params['spatial_modes']
            f.attrs['cn2_min'] = float(self.turb_params['cn2_min'])
            f.attrs['cn2_max'] = float(self.turb_params['cn2_max'])
            
            # Processing Loop
            buffer_intensity = []
            buffer_symbols = []
            buffer_cn2 = []
            
            # Use all available cores
            num_workers = mp.cpu_count()
            print(f"Starting pool with {num_workers} workers...")
            
            with mp.Pool(processes=num_workers, initializer=init_worker, initargs=(context,)) as pool:
                for result in tqdm(pool.imap(generate_single_sample, tasks, chunksize=10), total=num_samples):
                    intensity, symbols, metadata = result
                    
                    buffer_intensity.append(intensity)
                    buffer_symbols.append(np.stack([np.real(symbols), np.imag(symbols)], axis=-1))
                    buffer_cn2.append(metadata['cn2'])
                    
                    # Flush buffer if full
                    if len(buffer_intensity) >= chunk_size:
                        self._flush_buffer(f, dset_intensity, dset_symbols, dset_cn2, 
                                         buffer_intensity, buffer_symbols, buffer_cn2)
                        buffer_intensity = []
                        buffer_symbols = []
                        buffer_cn2 = []
            
            # Flush remaining
            if buffer_intensity:
                self._flush_buffer(f, dset_intensity, dset_symbols, dset_cn2, 
                                 buffer_intensity, buffer_symbols, buffer_cn2)
                
            # Update final count
            f.attrs['num_samples'] = dset_intensity.shape[0]
            
        print(f"\n✓ Dataset saved: {output_path}")
        print(f"  Samples: {num_samples}")
        print(f"  Size: {output_path.stat().st_size / (1024**2):.1f} MB")
        
        return output_path

    def _flush_buffer(self, f, dset_intensity, dset_symbols, dset_cn2, 
                     buf_int, buf_sym, buf_cn2):
        """Write buffer to HDF5 datasets."""
        n_new = len(buf_int)
        current_size = dset_intensity.shape[0]
        new_size = current_size + n_new
        
        # Resize
        dset_intensity.resize(new_size, axis=0)
        dset_symbols.resize(new_size, axis=0)
        dset_cn2.resize(new_size, axis=0)
        
        # Write
        dset_intensity[current_size:] = np.array(buf_int)
        dset_symbols[current_size:] = np.array(buf_sym)
        dset_cn2[current_size:] = np.array(buf_cn2)
        
        # Flush to disk
        f.flush()


def main():
    # Set start method to 'spawn' for better compatibility on macOS/Linux with complex objects
    # 'fork' is faster but can be problematic with some libraries
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="Generate FSO-OAM dataset for CNN training")
    parser.add_argument('--config', type=str, default='dataset/config.json',
                       help='Path to configuration file')
    parser.add_argument('--split', type=str, default='all', choices=['train', 'val', 'test', 'all'],
                       help='Which split to generate')
    parser.add_argument('--num-samples', type=int, default=None,
                       help='Override number of samples (for testing)')
    args = parser.parse_args()
    
    # Load config
    config_path = SCRIPT_DIR / args.config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize generator
    generator = DatasetGenerator(config_path)
    
    # Determine which splits to generate
    if args.split == 'all':
        splits = ['train', 'val', 'test']
    else:
        splits = [args.split]
    
    # Generate datasets
    for split in splits:
        if args.num_samples is not None:
            num_samples = args.num_samples
        else:
            num_samples = config['dataset_size'][split]
        
        generator.generate_dataset(num_samples, split=split)
    
    print(f"\n{'='*60}")
    print("✓ Dataset generation complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
