import numpy as np

def inspect_boundary():
    # Load Level 4 and Level 5 results
    f4 = "../../cnn_results_convnext_tiny_curriculum_lvl4_strong.npz"
    f5 = "../../cnn_results_convnext_tiny_curriculum_lvl5_extreme.npz"
    
    print("--- Level 4 (Strong) Tail ---")
    data4 = np.load(f4)
    cn2_4 = data4['cn2']
    ber_4 = data4['ber']
    # Print last 5 points
    for c, b in zip(cn2_4[-5:], ber_4[-5:]):
        print(f"Cn2: {c:.2e} | BER: {b:.4f}")
        
    print("\n--- Level 5 (Extreme) Head ---")
    data5 = np.load(f5)
    cn2_5 = data5['cn2']
    ber_5 = data5['ber']
    # Print first 5 points
    for c, b in zip(cn2_5[:5], ber_5[:5]):
        print(f"Cn2: {c:.2e} | BER: {b:.4f}")

if __name__ == "__main__":
    inspect_boundary()
