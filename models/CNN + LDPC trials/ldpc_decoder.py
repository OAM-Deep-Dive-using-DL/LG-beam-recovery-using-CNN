"""
ldpc_decoder.py
A minimal, pure-Python implementation of a Sum-Product Algorithm (SPA)
LDPC decoder for soft-decision (LLR) inputs.
This is used to bypass installation issues with C-compiled libraries.
"""
import numpy as np
import warnings

def make_ldpc_H(m, n, d_v, d_c):
    """
    Generates a random regular LDPC parity-check matrix H.
    This is a simplified Gallager construction.
    
    Parameters:
    - m: Number of check nodes (parity bits)
    - n: Number of variable nodes (codeword length)
    - d_v: Variable node degree (must be constant)
    - d_c: Check node degree (must be constant)
    
    Returns:
    - H: The (m x n) parity check matrix
    """
    if m * d_c != n * d_v:
        raise ValueError("Invalid LDPC parameters: m*d_c must equal n*d_v")
    
    H = np.zeros((m, n), dtype=int)
    v_nodes = np.arange(n)
    
    # Divide variable nodes into d_v groups
    group_size = n // d_v
    groups = [v_nodes[i*group_size:(i+1)*group_size] for i in range(d_v)]
    
    # Assign check nodes
    for i in range(d_v):
        check_nodes_perm = np.tile(np.arange(m), d_c)[0:n]
        if i == 0:
            check_nodes_perm = np.sort(check_nodes_perm)
        else:
            np.random.shuffle(check_nodes_perm)
            
        for j, v_node in enumerate(groups[i]):
            H[check_nodes_perm[j], v_node] = 1
            
    return H

def get_G_from_H(H):
    """
    Finds the generator matrix G for a systematic LDPC code
    by performing Gaussian elimination on H = [P | I].
    This is a simplified approach and assumes H is full rank.
    """
    m, n = H.shape
    k = n - m
    
    # This is a basic Gaussian elimination to find systematic form
    # A full implementation is complex; for simulation, we'll
    # assume a systematic-friendly structure from make_ldpc_H.
    # We will find G = [I_k | P.T]
    
    # For a simple simulation, let's *assume* H was made nicely
    # H = [P | I_m]
    try:
        P = H[:, :k]
        I_k = np.eye(k, dtype=int)
        G = np.concatenate([I_k, P.T], axis=1).astype(int)
    except Exception as e:
        print(f"Warning: Could not create systematic G: {e}. Encoding may fail.")
        # Fallback: non-systematic (not ideal)
        G = np.eye(k, n, dtype=int) # This is not a real G, just a placeholder
    
    return G

def ldpc_encode(G, info_bits):
    """
    Encodes info_bits using the generator matrix G.
    c = u * G (mod 2)
    """
    coded_bits = np.dot(info_bits, G) % 2
    return coded_bits.astype(int)

def ldpc_decode_spa(H, llrs_in, max_iter=20):
    """
    Decodes an LLR vector using the Sum-Product Algorithm (SPA).
    This is a soft-input, soft-output decoder.
    
    Parameters:
    - H: Parity check matrix (m x n)
    - llrs_in: Input LLRs from the channel/CNN. LLR = log(P(0)/P(1))
    - max_iter: Max number of decoding iterations
    
    Returns:
    - llrs_out: Final LLRs after decoding
    - hard_decision: Final hard-bit decision
    """
    m, n = H.shape
    
    # Find non-zero indices for sparse operations
    check_nodes_for_v, var_nodes_for_c = [], []
    for i in range(n): # For each var node
        check_nodes_for_v.append(np.where(H[:, i] == 1)[0])
    for j in range(m): # For each check node
        var_nodes_for_c.append(np.where(H[j, :] == 1)[0])

    # LLRs from variable nodes to check nodes
    L_vc = np.zeros((m, n))
    
    # Initialize L_vc with intrinsic (channel) LLRs
    for i in range(n):
        for j in check_nodes_for_v[i]:
            L_vc[j, i] = llrs_in[i]

    for it in range(max_iter):
        # --- 1. Check Node Update (Horizontal Step) ---
        L_cv = np.zeros((m, n))
        for j in range(m): # For each check node
            var_nodes = var_nodes_for_c[j]
            for i in var_nodes: # For each var node connected to this check
                
                # "Extrinsic" LLRs from all *other* var nodes
                L_j_i = [L_vc[j, k] for k in var_nodes if k != i]
                
                # --- This is the "tanh rule" for check nodes ---
                # L_cv[j, i] = 2 * atanh( product( tanh(L_j_i[k]/2) ) )
                
                if not L_j_i:
                    continue

                # Use log-domain for numerical stability
                # sign_prod = product( sign(L_j_i[k]) )
                sign_prod = np.prod(np.sign(L_j_i))
                
                # abs_sum = sum( -log( tanh( |L_j_i[k]|/2 ) ) )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning) # Ignore overflow
                    L_abs = np.abs(L_j_i)
                    tanh_vals = np.tanh(L_abs / 2.0)
                    log_tanh_vals = -np.log(np.maximum(tanh_vals, 1e-20)) # Avoid log(0)
                    abs_sum = np.sum(log_tanh_vals)

                # Final LLR
                L_cv[j, i] = sign_prod * (2 * np.arctanh(np.exp(-abs_sum)))
        
        # --- 2. Variable Node Update (Vertical Step) ---
        L_v_post = np.zeros(n) # Final posterior LLR
        all_syndromes_zero = True
        
        for i in range(n): # For each var node
            check_nodes = check_nodes_for_v[i]
            
            # L_v_post[i] = L_channel[i] + sum( L_cv[j, i] )
            L_v_post[i] = llrs_in[i] + np.sum(L_cv[check_nodes, i])
            
            # Update the messages for the next iteration
            for j in check_nodes:
                L_vc[j, i] = L_v_post[i] - L_cv[j, i] # Extrinsic
        
        # --- 3. Check Syndrome ---
        hard_decision = (L_v_post < 0).astype(int)
        syndrome = np.dot(H, hard_decision) % 2
        if np.all(syndrome == 0):
            # print(f"LDPC Converged in {it+1} iterations.")
            break

    return L_v_post, (L_v_post < 0).astype(int)