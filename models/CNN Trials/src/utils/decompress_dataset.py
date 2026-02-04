import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

def decompress_dataset(input_path, output_path):
    print(f"Decompressing {input_path} -> {output_path}")
    
    with h5py.File(input_path, 'r') as f_in:
        with h5py.File(output_path, 'w') as f_out:
            # Copy attributes
            for k, v in f_in.attrs.items():
                f_out.attrs[k] = v
                
            # Iterate over datasets
            for key in f_in.keys():
                dset_in = f_in[key]
                shape = dset_in.shape
                dtype = dset_in.dtype
                chunks = dset_in.chunks
                
                print(f"  Processing '{key}': {shape} {dtype}")
                
                # Create uncompressed dataset
                # We keep chunking for efficient partial I/O, but remove compression
                dset_out = f_out.create_dataset(
                    key, 
                    shape=shape, 
                    dtype=dtype, 
                    chunks=chunks, 
                    maxshape=dset_in.maxshape,
                    compression=None # DISABLE COMPRESSION
                )
                
                # Copy data in chunks to avoid RAM explosion
                chunk_size = 5000
                total = shape[0]
                
                for i in tqdm(range(0, total, chunk_size), desc=f"  Copying {key}"):
                    end = min(i + chunk_size, total)
                    data = dset_in[i:end]
                    dset_out[i:end] = data
                    
    print(f"✓ Decompression complete: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Decompress HDF5 dataset")
    parser.add_argument('--data_dir', type=Path, default=Path('../../data/dataset'))
    parser.add_argument('--dataset_name', type=str, default='config_fso')
    args = parser.parse_args()
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        input_file = args.data_dir / f"{args.dataset_name}_{split}.h5"
        # We will write to a temp file then rename, or just a new suffix
        output_file = args.data_dir / f"{args.dataset_name}_{split}_uncompressed.h5"
        
        if input_file.exists():
            decompress_dataset(input_file, output_file)
            
            # Backup original and rename uncompressed to main
            # backup_file = args.data_dir / f"{args.dataset_name}_{split}_compressed_backup.h5"
            # input_file.rename(backup_file)
            # output_file.rename(input_file)
            
            print(f"Created uncompressed version: {output_file}")
            print(f"To use it, rename it to {input_file} or update train script path.")
            
            # Auto-replace for user convenience?
            # Let's just swap them to fix the pipeline immediately
            backup_file = args.data_dir / f"{args.dataset_name}_{split}.h5.bak"
            if backup_file.exists():
                backup_file.unlink()
            input_file.rename(backup_file)
            output_file.rename(input_file)
            print(f"✓ Swapped in uncompressed file. Original backed up to .bak")
            
        else:
            print(f"Skipping {split}: File not found")

if __name__ == "__main__":
    main()
