
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
import warnings
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
CANONICAL_BASELINE = {
    "n_grid_sim": 512,
    "num_screens": 25,
    "pilot_power_ratio": 0.1,
    "snr_db": 35.0,
}

def init_worker(context):
    """Initialize worker process with shared data."""
    global WORKER_CONTEXT
    WORKER_CONTEXT.update(context)
    # Re-seed random number generator for each worker to ensure independence
    # Use process ID to facilitate unique seeds
    base_seed = context.get('random_seed', 42)
    np.random.seed(base_seed + os.getpid())

def _sample_snr_db(config):
    """Sample the SNR setting requested by the config."""
    aug_config = config.get('augmentation', {})
    snr_param = aug_config.get('snr_db_range', CANONICAL_BASELINE["snr_db"])

    if isinstance(snr_param, list):
        if len(snr_param) != 2:
            raise ValueError(f"snr_db_range must have two entries, got {snr_param}")
        snr_low, snr_high = map(float, snr_param)
        if snr_low > snr_high:
            raise ValueError(f"snr_db_range must be ordered [low, high], got {snr_param}")
        return float(np.random.uniform(snr_low, snr_high))

    return float(snr_param)

def _add_noise_like_pipeline(field_attenuated, aperture_mask, grid_info, config):
    """
    Add complex Gaussian field noise using the same per-pixel rule as pipeline.py.
    """
    aug_config = config.get('augmentation', {})
    if not aug_config.get('add_noise', False):
        return field_attenuated, {
            'snr_db': None,
            'noise_var_per_pixel': 0.0,
            'power_per_symbol': float(np.sum(np.abs(field_attenuated * aperture_mask) ** 2) * grid_info['delta']**2)
        }

    num_pixels_in_aperture = int(np.sum(aperture_mask))
    if num_pixels_in_aperture <= 0:
        num_pixels_in_aperture = 1

    dA = grid_info['delta'] ** 2
    field_in_aperture = field_attenuated * aperture_mask
    power_per_symbol = np.sum(np.abs(field_in_aperture) ** 2) * dA
    if power_per_symbol <= 0:
        return field_attenuated, {
            'snr_db': None,
            'noise_var_per_pixel': 0.0,
            'power_per_symbol': 0.0
        }

    target_snr_db = _sample_snr_db(config)
    avg_pixel_intensity = power_per_symbol / num_pixels_in_aperture
    snr_linear = 10 ** (target_snr_db / 10.0)
    noise_var_per_pixel = avg_pixel_intensity / snr_linear
    noise_std_per_pixel = np.sqrt(noise_var_per_pixel)

    noise = (noise_std_per_pixel / np.sqrt(2.0)) * (
        np.random.randn(*field_attenuated.shape) + 1j * np.random.randn(*field_attenuated.shape)
    )
    return field_attenuated + noise, {
        'snr_db': float(target_snr_db),
        'noise_var_per_pixel': float(noise_var_per_pixel),
        'power_per_symbol': float(power_per_symbol)
    }

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
    
    # 6. Aperture mask
    receiver_radius = sys_params['receiver_diameter'] / 2.0
    aperture_mask = (grid_info['R'] <= receiver_radius).astype(float)

    # 7. Add field noise using the canonical per-pixel pipeline rule,
    # then apply the receiver aperture exactly as the baseline does.
    E_rx_noisy, noise_metadata = _add_noise_like_pipeline(E_rx, aperture_mask, grid_info, config)
    E_rx_final = E_rx_noisy * aperture_mask
    intensity = np.abs(E_rx_final)**2

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
        'signal_power': noise_metadata['power_per_symbol'],
        'sample_snr_db': noise_metadata['snr_db'],
        'noise_var_per_pixel': noise_metadata['noise_var_per_pixel']
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
        self.data_format = self.config['data_format']
        self.augmentation = self.config.get('augmentation', {})
        self.dataset_size = self.config['dataset_size']

        self._validate_config()
        
        # Set Master Seed
        np.random.seed(self.config.get('random_seed', 42))
        
        # Initialize Shared Objects
        self.qpsk_modulator = QPSKModulator(symbol_energy=1.0)
        self._init_beams()
        self._generate_cn2_values()
        
        print(f"✓ Generator Initialized from {config_path.name}")
        print(f"  Cn2 Range: {self.turb_params['cn2_min']:.2e} -> {self.turb_params['cn2_max']:.2e}")
        self._report_baseline_alignment()

    def _validate_config(self):
        required_top_level = [
            'dataset_name', 'system_parameters', 'turbulence_parameters',
            'dataset_size', 'grid_parameters', 'data_format'
        ]
        missing = [key for key in required_top_level if key not in self.config]
        if missing:
            raise ValueError(f"Missing required config keys: {missing}")

        n_grid_sim = int(self.grid_params['n_grid_sim'])
        n_grid_output = int(self.grid_params['n_grid_output'])
        if n_grid_sim <= 0 or n_grid_output <= 0:
            raise ValueError("Grid sizes must be positive.")
        if n_grid_output > n_grid_sim:
            raise ValueError("n_grid_output cannot exceed n_grid_sim.")

        cn2_min = float(self.turb_params['cn2_min'])
        cn2_max = float(self.turb_params['cn2_max'])
        if cn2_min <= 0 or cn2_max <= 0:
            raise ValueError("Cn^2 bounds must be positive.")
        if cn2_min > cn2_max:
            raise ValueError("cn2_min must be <= cn2_max.")
        if int(self.turb_params['num_screens']) <= 0:
            raise ValueError("num_screens must be positive.")

        spatial_modes = self.sys_params.get('spatial_modes', [])
        if not spatial_modes:
            raise ValueError("system_parameters.spatial_modes cannot be empty.")

        input_shape = self.data_format.get('input_shape')
        if input_shape is not None:
            expected_shape = [n_grid_output, n_grid_output]
            if list(input_shape) != expected_shape:
                raise ValueError(
                    f"data_format.input_shape={input_shape} does not match n_grid_output={expected_shape}."
                )

        aug_enabled = bool(self.augmentation.get('enabled', False))
        rotation_range = float(self.augmentation.get('rotation_range', 0) or 0)
        translation_range = float(self.augmentation.get('translation_range', 0) or 0)
        multiple_realizations = int(self.augmentation.get('multiple_realizations', 1) or 1)
        if aug_enabled and (rotation_range != 0 or translation_range != 0 or multiple_realizations != 1):
            raise ValueError(
                "generate_dataset.py does not implement geometric augmentation or multiple realizations. "
                "Set augmentation.enabled=false or zero out those fields."
            )

        snr_param = self.augmentation.get('snr_db_range', CANONICAL_BASELINE['snr_db'])
        if isinstance(snr_param, list):
            if len(snr_param) != 2:
                raise ValueError("augmentation.snr_db_range must contain exactly two numbers.")
            if float(snr_param[0]) > float(snr_param[1]):
                raise ValueError("augmentation.snr_db_range must be ordered [low, high].")

        samples_per_cn2 = self.dataset_size.get('samples_per_cn2')
        if samples_per_cn2 not in (None, 'auto'):
            raise ValueError(
                "generate_dataset.py uses continuous Cn^2 sampling and does not support dataset_size.samples_per_cn2."
            )

        downsampling_method = self.grid_params.get('downsampling_method', 'bilinear')
        if downsampling_method not in ('bilinear', 'nearest'):
            raise ValueError("downsampling_method must be 'bilinear' or 'nearest'.")

    def _report_baseline_alignment(self):
        notes = []
        if int(self.grid_params['n_grid_sim']) != CANONICAL_BASELINE['n_grid_sim']:
            notes.append(f"n_grid_sim={self.grid_params['n_grid_sim']} (baseline {CANONICAL_BASELINE['n_grid_sim']})")
        if int(self.turb_params['num_screens']) != CANONICAL_BASELINE['num_screens']:
            notes.append(f"num_screens={self.turb_params['num_screens']} (baseline {CANONICAL_BASELINE['num_screens']})")
        pilot_ratio = float(self.sys_params.get('pilot_parameters', {}).get('power_ratio', 0.0))
        if abs(pilot_ratio - CANONICAL_BASELINE['pilot_power_ratio']) > 1e-12:
            notes.append(
                f"pilot power_ratio={pilot_ratio:.3f} (baseline {CANONICAL_BASELINE['pilot_power_ratio']:.3f})"
            )
        if notes:
            print("  Baseline alignment note:")
            for note in notes:
                print(f"    - {note}")
    
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

    def _sample_cn2_tasks(self, num_samples):
        if self.cn2_min == self.cn2_max:
            return [float(self.cn2_min)] * num_samples

        distribution = str(self.turb_params.get('cn2_distribution', 'logarithmic')).lower()
        weights_cfg = self.config.get('cn2_sampling_weights')

        def sample_range(low, high, count):
            if distribution.startswith('log'):
                return 10 ** np.random.uniform(np.log10(low), np.log10(high), count)
            if distribution.startswith('lin'):
                return np.random.uniform(low, high, count)
            raise ValueError(f"Unsupported cn2_distribution: {distribution}")

        if weights_cfg:
            regions = []
            weights = []
            for name, spec in weights_cfg.items():
                low, high = map(float, spec['range'])
                weight = float(spec['weight'])
                if low <= 0 or high <= 0 or low > high:
                    raise ValueError(f"Invalid cn2_sampling_weights range for '{name}': {spec['range']}")
                if low < self.cn2_min or high > self.cn2_max:
                    raise ValueError(
                        f"cn2_sampling_weights range {spec['range']} for '{name}' falls outside "
                        f"[{self.cn2_min}, {self.cn2_max}]"
                    )
                if weight < 0:
                    raise ValueError(f"Negative sampling weight for '{name}'.")
                regions.append((low, high))
                weights.append(weight)

            weight_sum = sum(weights)
            if weight_sum <= 0:
                raise ValueError("cn2_sampling_weights must contain at least one positive weight.")

            probs = np.array(weights, dtype=float) / weight_sum
            picks = np.random.choice(len(regions), size=num_samples, p=probs)
            sampled = np.empty(num_samples, dtype=float)
            for idx, (low, high) in enumerate(regions):
                mask = picks == idx
                if np.any(mask):
                    sampled[mask] = sample_range(low, high, int(np.sum(mask)))
            return sampled.tolist()

        return sample_range(float(self.cn2_min), float(self.cn2_max), num_samples).tolist()

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
        
        # Tasks: Sample Cn^2 according to the configured distribution.
        tasks = self._sample_cn2_tasks(num_samples)
        
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
            f.attrs['num_screens'] = int(self.turb_params['num_screens'])
            f.attrs['n_grid_sim'] = int(self.grid_params['n_grid_sim'])
            f.attrs['n_grid_output'] = int(self.grid_params['n_grid_output'])
            f.attrs['generator'] = 'generate_dataset.py'
            f.attrs['noise_model'] = 'pipeline_per_pixel_complex_field'

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

def resolve_config_path(config_arg: str) -> Path:
    """Resolve config path from absolute path or data/configs-relative name."""
    config_path = Path(config_arg)
    if config_path.exists():
        return config_path.resolve()

    candidate = DATA_DIR / 'configs' / Path(config_arg).name
    if candidate.exists():
        return candidate.resolve()

    raise FileNotFoundError(f"Config file not found: {config_arg}")


def main():
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, nargs='+', required=True,
                        help='One or more config files. Multiple configs run sequentially.')
    parser.add_argument('--split', type=str, default='all')
    parser.add_argument('--num-samples', type=int, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--workers', type=int, default=None, help="Number of worker processes per job")
    args = parser.parse_args()

    config_paths = []
    for config_arg in args.config:
        try:
            config_paths.append(resolve_config_path(config_arg))
        except FileNotFoundError as exc:
            print(f"Error: {exc}")
            sys.exit(1)

    splits = ['train', 'val', 'test'] if args.split == 'all' else [args.split]

    for config_path in config_paths:
        print(f"\n{'=' * 80}")
        print(f"Starting config: {config_path.name}")
        print(f"{'=' * 80}")
        generator = PhysicsGroundedDatasetGenerator(config_path)

        for split in splits:
            n = args.num_samples if args.num_samples is not None else generator.config['dataset_size'][split]
            out = Path(args.output_dir) / f"{generator.config['dataset_name']}_{split}.h5" if args.output_dir else None
            generator.generate_dataset(n, split=split, output_path=out, workers=args.workers)

if __name__ == "__main__":
    main()