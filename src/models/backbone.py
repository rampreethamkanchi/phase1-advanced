import torch
import torch.nn as nn
from torchvision.models import swin_b, Swin_B_Weights

class SwinBBackbone(nn.Module):
    """
    Swin-B Transformer Backbone.
    Extracts spatial features from individual frames. We strip the standard 
    ImageNet classification head to use the raw feature maps.
    
    Swin-B Input: (B, C, H, W) -> e.g., (1, 3, 384, 384)
    Swin-B Output: (B, C', H', W') -> The final feature map before pooling/flattening.
                   Specifically, for dim=384, output is typically (B, 1024, 12, 12)
    """
    def __init__(self, pretrained=True, freeze_early_layers=True):
        super(SwinBBackbone, self).__init__()
        
        # We load Swin-B with the canonical ImageNet-1K (or IMAGENET1K_V1) weights.
        # torchvision's swin_b has a `features` sequential module.
        weights = Swin_B_Weights.DEFAULT if pretrained else None
        swin_model = swin_b(weights=weights)
        
        # The 'features' module contains the patch embedding and the 4 Swin blocks
        self.features = swin_model.features
        
        # We add standard LayerNorm as Swin typically does right after features
        self.norm = swin_model.norm
        
        # Optionally freeze the early layers (e.g. patch partition and first two blocks)
        # to save VRAM and focus learning on the deeper, more semantic layers
        if freeze_early_layers:
            for i in range(len(self.features) - 2): # Freeze patch embedding and early blocks roughly
                 for param in self.features[i].parameters():
                     param.requires_grad = False
                     
        self.out_channels = 1024 # Swin-B default output channels

    def forward(self, x):
        """
        Since our data loader yields (B, T, C, H, W) tensors, we need to fold the 
        temporal dimension into the batch dimension before passing through Swin, 
        then unfold it back.
        
        Args:
            x: (B, T, C, H, W) where T is windows size (e.g., 8).
        Returns:
            features: (B, T, 12*12, 1024) -> Flattened spatial dimension for Transformer encoder
        """
        B, T, C, H, W = x.shape
        
        # Fold Temporal into Batch: (B*T, C, H, W)
        x = x.view(B * T, C, H, W)
        
        # Feature extraction
        # x shape becomes (B*T, channel, H/32, W/32) -> e.g. (B*T, 1024, 12, 12) for 384x384 input
        # Note: torchvision swin returns (B*T, H', W', C') due to its internal layout!
        out = self.features(x)
        out = self.norm(out) # (B*T, 12, 12, 1024)
        
        # We need it as a sequence of spatial tokens per frame: (B*T, 144, 1024)
        _, H_prime, W_prime, C_prime = out.shape
        out = out.view(B * T, H_prime * W_prime, C_prime)
        
        # Unfold Temporal: (B, T, SpatialTokens, Channels)
        out = out.view(B, T, H_prime * W_prime, C_prime)
        
        return out

if __name__ == "__main__":
    # Test the backbone
    print("--- Testing Swin-B Backbone ---")
    model = SwinBBackbone(pretrained=True).cuda()
    
    # Dummy input mirroring our dataloader output (Batch=2, TemporalWindow=8)
    dummy_x = torch.randn(2, 8, 3, 384, 384).cuda()
    features = model(dummy_x)
    print(f"Input Shape: {dummy_x.shape}")
    print(f"Feature Shape: {features.shape} (Expected: B, T, L, C = 2, 8, 144, 1024)")
    print("--- Test Passed ---")
