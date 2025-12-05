"""
Device detection and optimization for Apple Silicon (M1/M2/M3).

Automatically detects and configures the best available device:
- MPS (Metal Performance Shaders) for Apple Silicon
- CUDA for NVIDIA GPUs
- CPU as fallback
"""

import torch
import platform
import psutil


def get_device(prefer_mps=True, verbose=True):
    """
    Get the best available PyTorch device.
    
    Args:
        prefer_mps: If True, prefer MPS over CPU on Apple Silicon
        verbose: Print device information
    
    Returns:
        device: PyTorch device
        device_name: Human-readable device name
    """
    # Check for CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
    
    # Check for MPS (Apple Silicon)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and prefer_mps:
        device = torch.device('mps')
        device_name = f"MPS (Apple Silicon {platform.processor()})"
    
    # Fallback to CPU
    else:
        device = torch.device('cpu')
        device_name = f"CPU ({platform.processor()})"
    
    if verbose:
        print(f"🖥️  Using device: {device_name}")
        print_system_info()
    
    return device, device_name


def print_system_info():
    """Print system information for debugging."""
    print(f"\n📊 System Information:")
    print(f"  Platform: {platform.system()} {platform.release()}")
    print(f"  Processor: {platform.processor()}")
    print(f"  Python: {platform.python_version()}")
    print(f"  PyTorch: {torch.__version__}")
    
    # Memory info
    mem = psutil.virtual_memory()
    print(f"\n💾 Memory:")
    print(f"  Total: {mem.total / (1024**3):.1f} GB")
    print(f"  Available: {mem.available / (1024**3):.1f} GB")
    print(f"  Used: {mem.percent:.1f}%")
    
    # Device-specific info
    if torch.cuda.is_available():
        print(f"\n🎮 CUDA:")
        print(f"  Devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.1f} GB")
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print(f"\n🍎 MPS (Metal Performance Shaders):")
        print(f"  Available: Yes")
        print(f"  Built: {torch.backends.mps.is_built()}")
    
    print()


def optimize_for_device(model, device):
    """
    Optimize model for specific device.
    
    Args:
        model: PyTorch model
        device: Target device
    
    Returns:
        model: Optimized model
    """
    model = model.to(device)
    
    # Device-specific optimizations
    if device.type == 'cuda':
        # Enable cuDNN autotuner
        torch.backends.cudnn.benchmark = True
    
    elif device.type == 'mps':
        # MPS-specific optimizations
        # Note: MPS is still evolving, keep settings conservative
        pass
    
    return model


def get_optimal_batch_size(device, base_batch_size=64):
    """
    Get optimal batch size based on available memory.
    
    Args:
        device: PyTorch device
        base_batch_size: Base batch size for high-memory systems
    
    Returns:
        batch_size: Recommended batch size
    """
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024**3)
    
    if device.type == 'cuda':
        # Use GPU memory
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_mem_gb >= 16:
            return base_batch_size
        elif gpu_mem_gb >= 8:
            return base_batch_size // 2
        else:
            return base_batch_size // 4
    
    elif device.type == 'mps':
        # MPS shares system memory
        # Conservative for 8GB systems
        if available_gb >= 6:
            return 32  # Safe for 8GB M3
        elif available_gb >= 4:
            return 16
        else:
            return 8
    
    else:  # CPU
        # Very conservative for CPU
        return min(16, base_batch_size // 4)


def get_num_workers(device):
    """
    Get optimal number of data loader workers.
    
    Args:
        device: PyTorch device
    
    Returns:
        num_workers: Recommended number of workers
    """
    cpu_count = psutil.cpu_count(logical=False) or 4
    
    if device.type == 'cuda':
        # More workers for GPU (data loading is bottleneck)
        return min(8, cpu_count)
    
    elif device.type == 'mps':
        # Moderate workers for MPS (shares memory with GPU)
        return min(2, cpu_count // 2)
    
    else:  # CPU
        # Fewer workers for CPU (computation is bottleneck)
        return min(2, cpu_count // 2)


def check_memory_usage():
    """Check current memory usage."""
    mem = psutil.virtual_memory()
    print(f"Memory: {mem.percent:.1f}% used ({mem.used / (1024**3):.1f} / {mem.total / (1024**3):.1f} GB)")
    
    if mem.percent > 90:
        print("⚠️  Warning: Memory usage is high!")
    
    return mem.percent


def clear_memory(device):
    """Clear memory cache."""
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()
    
    import gc
    gc.collect()


if __name__ == "__main__":
    # Test device detection
    print("="*60)
    print("Device Detection Test")
    print("="*60)
    
    device, device_name = get_device(verbose=True)
    
    print(f"\n🎯 Recommended Settings for {device_name}:")
    print(f"  Batch size: {get_optimal_batch_size(device)}")
    print(f"  Num workers: {get_num_workers(device)}")
    
    # Test tensor operations
    print(f"\n🧪 Testing tensor operations on {device}...")
    x = torch.randn(100, 100).to(device)
    y = torch.randn(100, 100).to(device)
    z = torch.matmul(x, y)
    print(f"  ✓ Matrix multiplication successful")
    print(f"  Result shape: {z.shape}")
    
    # Memory check
    print(f"\n💾 Memory check:")
    check_memory_usage()
    
    print("\n✅ Device detection test complete!")
