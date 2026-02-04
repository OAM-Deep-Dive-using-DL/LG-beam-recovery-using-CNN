
import time
import torch
import sys
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models.model import FSOModel
from utils.dataset import FSODataset
from torch.utils.data import DataLoader

def benchmark(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Data Loading Benchmark
    print(f"\n--- Benchmarking Data Loading (Workers={args.workers}, Batch={args.batch_size}) ---")
    dataset = FSODataset(args.data_path, split='train', augment=args.augment)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, 
                        num_workers=args.workers, persistent_workers=True)
    
    start_time = time.time()
    count = 0
    max_batches = 50
    
    for _ in tqdm(loader, total=max_batches, desc="Data Loading"):
        count += 1
        if count >= max_batches:
            break
            
    end_time = time.time()
    data_throughput = (count * args.batch_size) / (end_time - start_time)
    print(f"Data Throughput: {data_throughput:.2f} samples/sec")
    
    # 2. Model Compute Benchmark (Synthetic Data)
    print(f"\n--- Benchmarking Model Compute ({args.backbone}) ---")
    model = FSOModel(n_modes=8, backbone_name=args.backbone).to(device)
    model.train()
    
    # Synthetic Input
    inputs = torch.randn(args.batch_size, 1, 128, 128).to(device)
    
    # Warmup
    for _ in range(10):
        with torch.amp.autocast('mps'):
            _ = model(inputs)
            
    torch.mps.synchronize()
    start_time = time.time()
    
    # Run loop
    optimizer = torch.optim.AdamW(model.parameters())
    scaler = torch.amp.GradScaler('mps')
    criterion = torch.nn.MSELoss()
    target = torch.randn(args.batch_size, 8, 2).to(device)
    
    for _ in tqdm(range(max_batches), desc="Model Compute"):
        optimizer.zero_grad()
        with torch.amp.autocast('mps'):
            syms, _ = model(inputs)
            loss = criterion(syms, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.mps.synchronize() # Wait for GPU
        
    end_time = time.time()
    compute_throughput = (max_batches * args.batch_size) / (end_time - start_time)
    print(f"Compute Throughput: {compute_throughput:.2f} samples/sec")
    
    print(f"\n--- Analysis ---")
    if data_throughput < compute_throughput:
        print(f"BOTTLENECK: DATA LOADING is {compute_throughput/data_throughput:.1f}x slower than model.")
    else:
        print(f"BOTTLENECK: COMPUTE (Model) is {data_throughput/compute_throughput:.1f}x slower than data.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/dataset/config_fso_train.h5')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--backbone', type=str, default='convnext_tiny')
    parser.add_argument('--augment', action='store_true')
    args = parser.parse_args()
    
    benchmark(args)
