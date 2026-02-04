import torch
import torch.nn as nn

class PolarLoss(nn.Module):
    """
    Polar Loss function for Phase-Sensitive Signal Recovery.
    
    Decouples the loss into:
    1. Magnitude Loss (MSE): Ensures correct energy/amplitude recovery.
    2. Phase Loss (Cosine Distance): Ensures correct phase angle recovery.
    
    Formula:
        L = alpha * MSE(mag_pred, mag_true) + beta * (1 - cos(phase_pred - phase_true))
    """
    def __init__(self, alpha=1.0, beta=1.0, epsilon=1e-8):
        super(PolarLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.mse = nn.MSELoss()

    def forward(self, pred_complex, target_complex):
        """
        Args:
            pred_complex: (B, M, 2) or (B, M) complex tensor
            target_complex: (B, M, 2) or (B, M) complex tensor
        """
        # Ensure we are working with complex tensors
        if pred_complex.shape[-1] == 2 and pred_complex.dim() == 3:
            pred_c = torch.view_as_complex(pred_complex)
        else:
            pred_c = pred_complex
            
        if target_complex.shape[-1] == 2 and target_complex.dim() == 3:
            target_c = torch.view_as_complex(target_complex)
        else:
            target_c = target_complex

        # 1. Magnitude Loss (Amplitude)
        mag_pred = torch.abs(pred_c)
        mag_target = torch.abs(target_c)
        loss_mag = self.mse(mag_pred, mag_target)

        # 2. Phase Loss (Angle)
        # Cosine Similarity: <p, t> / (|p|*|t|)
        # Technically: cos(theta_p - theta_t) = Re(p * conj(t)) / (|p|*|t|)
        
        # Dot product per mode
        dot_product = torch.real(pred_c * torch.conj(target_c))
        denominator = mag_pred * mag_target + self.epsilon
        
        # Cosine similarity (1.0 is perfect alignment, -1.0 is opposite)
        cosine_sim = dot_product / denominator
        
        # We want to minimize distance, so Loss = 1 - Cosine
        # Range: [0, 2]
        loss_phase = torch.mean(1.0 - cosine_sim)

        return self.alpha * loss_mag + self.beta * loss_phase
