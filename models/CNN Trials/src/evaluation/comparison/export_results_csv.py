import numpy as np
import pandas as pd
from pathlib import Path

def export_csv():
    project_root = Path("/Users/srivatsadavuluri/Developer/FSO beam recovery")
    
    files = [
        ("cnn_results_convnext_tiny_curriculum_lvl1_ideal.npz", "Level 1 (Ideal)"),
        ("cnn_results_convnext_tiny_curriculum_lvl2_weak.npz", "Level 2 (Weak)"),
        ("cnn_results_convnext_tiny_curriculum_lvl3_moderate.npz", "Level 3 (Moderate)"),
        ("cnn_results_convnext_tiny_curriculum_lvl4_strong.npz", "Level 4 (Strong)"),
        ("cnn_results_convnext_tiny_curriculum_lvl5_extreme.npz", "Level 5 (Extreme)")
    ]
    
    all_rows = []
    
    print("Loading data...")
    for fname, label in files:
        fpath = project_root / fname
        if not fpath.exists():
            print(f"Warning: File not found: {fpath}")
            continue
            
        try:
            data = np.load(fpath)
            cn2 = data['cn2']
            ber = data['ber']
            
            # Create rows
            for c, b in zip(cn2, ber):
                all_rows.append({
                    "Cn2": c,
                    "BER": b,
                    "Dataset": label
                })
        except Exception as e:
            print(f"Error reading {fname}: {e}")

    # Create DataFrame
    df = pd.DataFrame(all_rows)
    
    # Sort by Cn2
    df = df.sort_values(by="Cn2")
    
    # Save to CSV
    output_path = project_root / "cumulative_results.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\nSuccessfully exported {len(df)} rows to: {output_path}")
    
    # Highlight the outlier
    outlier = df[np.isclose(df['Cn2'], 1e-14, atol=1e-16)]
    if not outlier.empty:
        print("\n--- Outlier Detected at 1e-14 ---")
        print(outlier.to_string(index=False))
        print("---------------------------------")

if __name__ == "__main__":
    export_csv()
