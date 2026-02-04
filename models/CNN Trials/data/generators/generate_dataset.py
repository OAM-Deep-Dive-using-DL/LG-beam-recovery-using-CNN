
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
# from functools import partial # Unused

# -----------------------------------------------------------------------------
# PATH OBSESSION: Ensure we can confirm exactly where we are
# -----------------------------------------------------------------------------
# This script is at: models/CNN Trials/data/generators/generate_dataset.py
SCRIPT_PATH = Path(__file__).resolve()
GENERATORS_DIR = SCRIPT_PATH.parent                # .../data/generators
DATA_DIR = GENERATORS_DIR.parent                   # .../data
CNN_TRIALS_DIR = DATA_DIR.parent                   # .../CNN Trials

# Add physics modules to path
# Physics is at: models/CNN Trials/physics
PHYSICS_DIR = CNN_TRIALS_DIR / 'physics'
sys.path.insert(0, str(PHYSICS_DIR))

try:
    from lgBeam import LaguerreGaussianBeam
    from turbulence import create_multi_layer_screens, apply_multi_layer_turbulence
    from fsplAtmAttenuation import calculate_kim_attenuation
    from encoding import QPSKModulator 
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import physics modules from {PHYSICS_DIR}")
    print(f"Error: {e}")
    sys.exit(1)

# Global storage for worker processes to share read-only data
WORKER_CONTEXT = {}

def init_worker(context):
    """Initialize worker process with shared data."""
    global WORKER_CONTEXT
    WORKER_CONTEXT.update(context)
    # Re-seed random number generator for each worker to ensure independence
    # Use process ID to facilitate unique seeds
    base_seed = context.get('random_seed', 42)
    np.random.seed(base_seed + os.getpid())

def _add_noise_proper(intensity, config, signal_power):
    """
    Add Gaussian noise matching pipeline.py methodology.
    Supports both Fixed SNR and Variable SNR ranges.
    """
    aug_config = config.get('augmentation', {})
    if not aug_config.get('add_noise', False):
        return intensity

    if signal_power <= 0:
        return intensity
    
    # -------------------------------------------------------------------------
    # SUPERIORITY FEATURE: Variable SNR Support
    # -------------------------------------------------------------------------
    snr_param = aug_config.get('snr_db_range', 35) # Default 35dB if missing
    
    if isinstance(snr_param, list):
        # Continuous sampling from range [min, max]
        target_snr_db = np.random.uniform(snr_param[0], snr_param[1])
    else:
        # Fixed Value
        target_snr_db = float(snr_param)

    # Calculate Noise Power
    # SNR_dB = 10 * log10(P_signal / P_noise)
    # => P_noise = P_signal / 10^(SNR/10)
    noise_power = signal_power / (10**(target_snr_db/10.0))
    noise_std = np.sqrt(noise_power)
    
    noise = np.random.normal(0, noise_std, intensity.shape)
    noisy_intensity = intensity + noise
    
    # Clip to non-negative (Physical Intensity Constraint)
    return np.maximum(noisy_intensity, 0.0)

def _zoom_to_aperture(intensity, grid_info, receiver_diameter):
    """
    Crop the intensity field to the receiver aperture before downsampling.
    Matches pipeline.py Smart Zoom implementation.
    """
    D_rx = receiver_diameter
    delta = grid_info['delta']
    
    # Calculate radius in pixels
    # D_rx is diameter. Radius = D_rx/2
    radius_pixels = int(np.ceil((D_rx / 2.0) / delta))
    
    center_idx = grid_info['N'] // 2
    
    # Extract indices centered on the grid
    i_min = max(0, center_idx - radius_pixels)
    i_max = min(grid_info['N'], center_idx + radius_pixels)
    j_min = max(0, center_idx - radius_pixels)
    j_max = min(grid_info['N'], center_idx + radius_pixels)
    
    # Crop
    intensity_cropped = intensity[i_min:i_max, j_min:j_max]
    return intensity_cropped

def generate_single_sample_physics(cn2_value):
    """
    Worker function to generate a single sample.
    """
    ctx = WORKER_CONTEXT
    sys_params = ctx['sys_params']
    turb_params = ctx['turb_params']
    grid_params = ctx['grid_params']
    grid_info = ctx['grid_info']
    basis_fields = ctx['basis_fields']
    lg_beams_keys = ctx['lg_beams_keys']
    max_m2_beam = ctx['max_m2_beam']
    config = ctx['config']
    qpsk_modulator = ctx['qpsk_modulator']
    
    # 1. Generate QPSK Symbols
    n_modes = len(sys_params['spatial_modes'])
    bits = np.random.randint(0, 2, n_modes * 2)  # 2 bits per mode
    symbols = qpsk_modulator.modulate(bits)      # Complex symbols
    
    # 2. Multiplex Transmit Field (E_tx)
    E_tx = np.zeros((grid_info['N'], grid_info['N']), dtype=complex)
    
    # Power Allocation
    total_power = sys_params['p_tx_total']
    pilot_params = sys_params.get('pilot_parameters', {'enabled': False})
    
    if pilot_params['enabled']:
        pilot_ratio = pilot_params.get('power_ratio', 0.1)
        pilot_power = total_power * pilot_ratio
        signal_power = total_power * (1 - pilot_ratio)
        
        # Add Pilot
        if 'pilot_field' in ctx and ctx['pilot_field'] is not None:
             E_tx += ctx['pilot_field'] * np.sqrt(pilot_power)
    else:
        signal_power = total_power

    # Add Data Modes
    power_per_mode = signal_power / n_modes
    scale_per_mode = np.sqrt(power_per_mode)
    
    for i, mode_key in enumerate(lg_beams_keys):
        # basis_fields are unit-power. Scale by amplitude (sqrt(P)) and phase (symbol)
        E_tx += basis_fields[mode_key] * symbols[i] * scale_per_mode
    
    # 3. Create Turbulence Screens
    # Note: We pass the specific cn2_value for this sample (Continuous Sampling)
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
    
    # 4. Propagate (Split-Step)
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
    
    # 5. Atmospheric Attenuation
    visibility_km = 23.0 
    alpha_dBkm = calculate_kim_attenuation(sys_params['wavelength'] * 1e9, visibility_km)
    L_atm_dB = alpha_dBkm * (sys_params['distance'] / 1000.0)
    amplitude_loss = 10**(-L_atm_dB / 20.0)
    E_rx = E_rx * amplitude_loss
    
    # 6. Aperture Mask
    receiver_radius = sys_params['receiver_diameter'] / 2.0
    aperture_mask = (grid_info['R'] <= receiver_radius).astype(float)
    E_rx = E_rx * aperture_mask
    
    # 7. Compute Intensity & Add Noise
    intensity = np.abs(E_rx)**2
    # Calculate signal power actually hitting the aperture
    signal_power_in_aperture = np.sum(intensity) * grid_info['delta']**2
    
    intensity = _add_noise_proper(intensity, config, signal_power_in_aperture)
    
    # 8. Smart Zoom & Downsample
    intensity_zoomed = _zoom_to_aperture(intensity, grid_info, sys_params['receiver_diameter'])
    
    n_out = grid_params['n_grid_output']
    scale_factor = n_out / intensity_zoomed.shape[0] # Should be <= 1 (downsampling)
    
    order = 1 if grid_params['downsampling_method'] == 'bilinear' else 0
    intensity_downsampled = zoom(intensity_zoomed, scale_factor, order=order)
    
    # 9. Normalization (Peak)
    if config['data_format']['normalize_input']:
        if config['data_format']['normalization_method'] == 'per_sample':
            max_val = np.max(intensity_downsampled)
            if max_val > 0:
                intensity_downsampled /= max_val
    
    # Metadata
    metadata = {
        'cn2': cn2_value,
        'distance': sys_params['distance'],
        'wavelength': sys_params['wavelength'],
        'attenuation_dB': L_atm_dB,
        'signal_power': float(signal_power_in_aperture),
        'snr_db_range': config.get('augmentation', {}).get('snr_db_range', 'unknown')
    }
    
    return intensity_downsampled, symbols, metadata

class PhysicsGroundedDatasetGenerator:
    """Generate physics-grounded FSO-OAM dataset."""
    
    def __init__(self, config_path):
        config_path = Path(config_path).resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Extract parameters
        self.sys_params = self.config['system_parameters']
        self.turb_params = self.config['turbulence_parameters']
        self.grid_params = self.config['grid_parameters']
        
        # Set Master Seed
        np.random.seed(self.config.get('random_seed', 42))
        
        # Initialize Shared Objects
        self.qpsk_modulator = QPSKModulator(symbol_energy=1.0)
        self._init_beams()
        self._generate_cn2_values()
        
        print(f"✓ Generator Initialized from {config_path.name}")
        print(f"  Cn2 Range: {self.turb_params['cn2_min']:.2e} -> {self.turb_params['cn2_max']:.2e}")
    
    def _init_beams(self):
        self.lg_beams = {}
        for mode in self.sys_params['spatial_modes']:
            p, l = mode
            beam = LaguerreGaussianBeam(
                p=p, l=l,
                wavelength=self.sys_params['wavelength'],
                w0=self.sys_params['w0']
            )
            self.lg_beams[tuple(mode)] = beam
        
        # Max M^2 determines the widest beam -> grid size
        self.max_m2_beam = max(self.lg_beams.values(), key=lambda b: b.M_squared)
    
    def _generate_cn2_values(self):
        # We store boundaries for continuous sampling logic
        self.cn2_min = self.turb_params['cn2_min']
        self.cn2_max = self.turb_params['cn2_max']

    def _setup_grid(self):
        distance = self.sys_params['distance']
        beam_size_at_rx = self.max_m2_beam.physical_beam_radius(distance)
        
        D = self.grid_params['oversampling'] * 6 * beam_size_at_rx
        N = self.grid_params['n_grid_sim']
        delta = D / N
        
        x = np.linspace(-D/2, D/2, N)
        y = np.linspace(-D/2, D/2, N)
        X, Y = np.meshgrid(x, y, indexing='ij')
        R = np.sqrt(X**2 + Y**2)
        PHI = np.arctan2(Y, X)
        
        return {'D': D, 'delta': delta, 'N': N, 'x': x, 'y': y, 'R': R, 'PHI': PHI}
    
    def _generate_basis_fields(self, grid_info):
        dA = grid_info['delta']**2
        basis_fields = {}
        for mode_key, beam in self.lg_beams.items():
            E_basis = beam.generate_beam_field(grid_info['R'], grid_info['PHI'], 0)
            energy = np.sum(np.abs(E_basis)**2) * dA
            if energy > 0: E_basis /= np.sqrt(energy)
            basis_fields[mode_key] = E_basis
        return basis_fields

    def _generate_pilot_field(self, grid_info):
        pilot_params = self.sys_params.get('pilot_parameters', {'enabled': False})
        if not pilot_params['enabled']: return None
        p, l = pilot_params.get('mode', [0, 0])
        beam = LaguerreGaussianBeam(p=p, l=l, wavelength=self.sys_params['wavelength'], w0=self.sys_params['w0'])
        E_pilot = beam.generate_beam_field(grid_info['R'], grid_info['PHI'], 0)
        dA = grid_info['delta']**2
        energy = np.sum(np.abs(E_pilot)**2) * dA
        if energy > 0: E_pilot /= np.sqrt(energy)
        return E_pilot
    
    def generate_dataset(self, num_samples, split='train', output_path=None, workers=None):
        if output_path is None:
            dataset_name = self.config['dataset_name']
            output_path = DATA_DIR / f'{dataset_name}_{split}.h5'
        
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Generating {split.upper()} -> {output_path.name}")
        
        # Grid Setup
        grid_info = self._setup_grid()
        basis_fields = self._generate_basis_fields(grid_info)
        pilot_field = self._generate_pilot_field(grid_info)
        
        # Worker Context
        context = {
            'sys_params': self.sys_params,
            'turb_params': self.turb_params,
            'grid_params': self.grid_params,
            'grid_info': grid_info,
            'basis_fields': basis_fields,
            'pilot_field': pilot_field,
            'lg_beams_keys': list(self.lg_beams.keys()),
            'max_m2_beam': self.max_m2_beam,
            'config': self.config,
            'random_seed': self.config.get('random_seed', 42),
            'qpsk_modulator': self.qpsk_modulator
        }
        
        # Tasks: Continuous Log-Uniform Sampling of Cn2
        log_min = np.log10(self.cn2_min)
        log_max = np.log10(self.cn2_max)
        log_cn2_samples = np.random.uniform(log_min, log_max, num_samples)
        tasks = (10**log_cn2_samples).tolist()
        
        # HDF5
        n_out = self.grid_params['n_grid_output']
        n_modes = len(self.sys_params['spatial_modes'])
        chunk_size = 1000
        
        with h5py.File(output_path, 'w') as f:
            dset_intensity = f.create_dataset('intensity', shape=(0, n_out, n_out), 
                                            maxshape=(None, n_out, n_out), dtype=np.float32, chunks=(100, n_out, n_out), compression='gzip')
            dset_symbols = f.create_dataset('symbols', shape=(0, n_modes, 2),
                                          maxshape=(None, n_modes, 2), dtype=np.float32, chunks=(100, n_modes, 2), compression='gzip')
            dset_cn2 = f.create_dataset('cn2', shape=(0,), maxshape=(None,), dtype=np.float32)
            
            # Metadata Attributes
            f.attrs['split'] = split
            f.attrs['n_modes'] = n_modes
            f.attrs['input_shape'] = [n_out, n_out]
            f.attrs['spatial_modes'] = self.sys_params['spatial_modes']
            f.attrs['cn2_min'] = float(self.cn2_min)
            f.attrs['cn2_max'] = float(self.cn2_max)

            # Processing
            buffer_intensity = []
            buffer_symbols = []
            buffer_cn2 = []
            
            # Cap workers to avoid memory separation issues, but use sufficient count
            if workers is None:
                num_workers = min(mp.cpu_count(), 8)
            else:
                num_workers = workers
            
            print(f"Starting pool with {num_workers} workers...")
            
            with mp.Pool(processes=num_workers, initializer=init_worker, initargs=(context,)) as pool:
                # Use default chunksize or optimize
                for result in tqdm(pool.imap(generate_single_sample_physics, tasks, chunksize=10), total=num_samples, leave=False):
                    int_map, syms, meta = result
                    
                    # Split complex to real/imag for storage
                    syms_iq = np.stack([np.real(syms), np.imag(syms)], axis=-1)
                    
                    buffer_intensity.append(int_map)
                    buffer_symbols.append(syms_iq)
                    buffer_cn2.append(meta['cn2'])
                    
                    if len(buffer_intensity) >= chunk_size:
                        self._flush_buffer(f, dset_intensity, dset_symbols, dset_cn2, buffer_intensity, buffer_symbols, buffer_cn2)
                        buffer_intensity, buffer_symbols, buffer_cn2 = [], [], []

            if buffer_intensity:
                self._flush_buffer(f, dset_intensity, dset_symbols, dset_cn2, buffer_intensity, buffer_symbols, buffer_cn2)
            
            f.attrs['num_samples'] = dset_intensity.shape[0]

        return output_path

    def _flush_buffer(self, f, dset_int, dset_sym, dset_cn2, b_int, b_sym, b_cn2):
        n = len(b_int)
        curr = dset_int.shape[0]
        new = curr + n
        
        dset_int.resize(new, axis=0)
        dset_sym.resize(new, axis=0)
        dset_cn2.resize(new, axis=0)
        
        dset_int[curr:] = np.array(b_int)
        dset_sym[curr:] = np.array(b_sym)
        dset_cn2[curr:] = np.array(b_cn2)
        f.flush()

def main():
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--split', type=str, default='all')
    parser.add_argument('--num-samples', type=int, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--workers', type=int, default=None, help="Number of worker processes per job")
    args = parser.parse_args()
    
    # Robust Config Loading
    # 1. Try absolute/relative as given
    config_path = Path(args.config)
    if not config_path.exists():
         # 2. Try relative to CNN Trials/ (Common case since user runs from root)
         # If args.config is "data/configs/..." and we are in root, it should have been found.
         # But if user provided just "configs/...", we check relative to script parent path structure.
         # Script is in .../data/generators/
         # Configs are in .../data/configs/
         
         # Try .../data/configs/NAME
         candidate = DATA_DIR / 'configs' / Path(args.config).name
         if candidate.exists():
             config_path = candidate
         else:
             print(f"Error: Config file not found: {args.config}")
             sys.exit(1)
             
    generator = PhysicsGroundedDatasetGenerator(config_path)
    
    splits = ['train', 'val', 'test'] if args.split == 'all' else [args.split]
    
    for split in splits:
        n = args.num_samples if args.num_samples is not None else generator.config['dataset_size'][split]
        out = Path(args.output_dir) / f"{generator.config['dataset_name']}_{split}.h5" if args.output_dir else None
        generator.generate_dataset(n, split=split, output_path=out, workers=args.workers)

if __name__ == "__main__":
    main()