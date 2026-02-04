
import os
import sys
import subprocess
import shutil
from pathlib import Path
import time

# Configuration
START_LEVEL = 2 # Start from Level 2 (Weak Turbulence)

LEVELS = [
    ("curriculum_lvl1_ideal", 100),
    ("curriculum_lvl2_weak", 100),
    ("curriculum_lvl3_moderate", 100),
    ("curriculum_lvl4_strong", 100),
    ("curriculum_lvl5_extreme", 100)
]

BACKBONE = "convnext_tiny"
DATA_DIR = Path("data") # Files are in data/ derived from recent ls
PYTHON = sys.executable

def run_command(cmd):
    print(f"\n[Curriculum] Executing: {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        print(f"❌ Command failed with return code {ret}")
        sys.exit(ret)

def main():
    print("=== Starting FSO-OAM Curriculum Training ===")
    
    # Ensure we are in project root (where src is)
    if not (Path.cwd() / "src").exists():
        print("Error: Must run from project root (where src/ exists)")
        sys.exit(1)

    best_model_name = f"best_model_{BACKBONE}.pth"
    
    for i, (dataset_name, epochs) in enumerate(LEVELS):
        level_idx = i + 1
        
        if level_idx < START_LEVEL:
            print(f"Skipping Level {level_idx}: {dataset_name} (START_LEVEL={START_LEVEL})")
            continue
            
        print(f"\n\n{'='*60}")
        print(f" LEVEL {level_idx}: {dataset_name}")
        print(f"{'='*60}")
        
        # Determine flags
        flags = f"--dataset_name {dataset_name} --epochs {epochs} --backbone {BACKBONE} --data_dir {DATA_DIR} --workers 4 --batch_size 64"
        
        if level_idx > 1:
            # Finetune from previous best
            # Ensure the 'best_model.pth' currently exists and is the one from the previous step
            if not Path(best_model_name).exists():
                print(f"❌ Error: Previous model {best_model_name} not found for finetuning!")
                sys.exit(1)
                
            flags += " --finetune --lr 5e-5"
            print(f"-> Finetuning from previous level (LR=5e-5)...")
        else:
            # First level: Standard training
            flags += " --lr 1e-4"
            print(f"-> Training from scratch (LR=1e-4)...")

        # Run Training
        cmd = f'"{PYTHON}" src/training/train.py {flags}'
        run_command(cmd)
        
        # Post-Training Management
        if Path(best_model_name).exists():
            # 1. Archive this level's model
            archive_name = f"model_lvl{level_idx}_{dataset_name}.pth"
            shutil.copy(best_model_name, archive_name)
            print(f"✓ Archived model to: {archive_name}")
            
            # The 'best_model.pth' stays in place to be the starting point for the next level
        else:
            print(f"⚠️ Warning: No best model found after Level {level_idx}. Training might have failed.")
            sys.exit(1)
            
    print("\n\n=== Curriculum Training Complete ===")
    print("Final Model: best_model_convnext_tiny.pth (corresponds to Level 5)")
    print("Archived Models: model_lvl*.pth")

if __name__ == "__main__":
    main()
