import torch
import torch.nn as nn
import torch.nn.functional as F

class MutualChannelLoss(nn.Module):
    """
    Mutual-Channel Loss (MCL).
    
    Aims to decorrelate feature channels in the spatial feature maps.
    This encourages different channels to focus on different highly discriminative
    regions (e.g. one focuses on the tool tip, another on the shaft, another on tissue).
    Helps prevent the network from collapsing all attention onto a single dominant feature.
    """
    def __init__(self, alpha=1.0):
        super(MutualChannelLoss, self).__init__()
        self.alpha = alpha

    def forward(self, features):
        """
        Args:
            features: (B, T, L_spatial, C) or (B, C, H, W). 
                      We expect (B, T, L, C) from our Swin Backbone.
        """
        with torch.amp.autocast('cuda', enabled=False):
            features = features.float()
            
            if features.dim() == 4:
                B, T, L, C = features.shape
                # Combine B and T for spatial analysis
                # We want to analyze correlation across the sequence length (L spatial tokens)
                features = features.view(B * T, L, C)
            elif features.dim() == 3:
                B, L, C = features.shape
            else:
                raise ValueError(f"Unexpected feature dimension for MCL: {features.shape}")
                
            # We want to minimize the cosine similarity between different channel vectors 
            # across the spatial dimension.
            # Normalize channels across spatial locations
            # Channel feature vector v_c is of size (B, L)
            
            # (B, L, C) -> (B, C, L)
            features = features.transpose(1, 2)
            
            # Normalize each channel vector
            features_norm = F.normalize(features, p=2, dim=2)
            
            # Compute Cosine Similarity Matrix (C x C) representing correlation between channels
            # (B, C, L) @ (B, L, C) -> (B, C, C)
            cc_sim = torch.bmm(features_norm, features_norm.transpose(1, 2))
            
            # We want to minimize the off-diagonal elements
            # Identity matrix represents perfect self-correlation
            I = torch.eye(C, device=features.device).expand(features.shape[0], C, C)
            
            # Sum of absolute off-diagonal similarities
            loss = torch.sum(torch.abs(cc_sim - I)) / (features.shape[0] * C * (C - 1) + 1e-8)
            
            # Optional: Add a penalty for channels that are entirely zero (inactive)
            # by checking the L2 norm before normalization.
            
        return self.alpha * loss

if __name__ == "__main__":
    print("--- Testing Mutual Channel Loss ---")
    loss_fn = MutualChannelLoss()
    
    # Dummy feature tensor reflecting Swin-B output: (B, T, SpatialTokens, C)
    dummy_feats = torch.randn(2, 8, 144, 1024)
    loss = loss_fn(dummy_feats)
    
    print(f"MCL Loss output: {loss.item()}")
    print("--- Test Passed ---")
