import torch
import torch.nn as nn
from .backbone import SwinBBackbone
from .t_encoder import BandedCausalTemporalEncoder
from .query_decoder import BipartiteBindingDecoder
from .refiner import DenoisingSemanticRefiner

class TriDiffTransformer(nn.Module):
    """
    Tri-Diff-Transformer (TDT).
    Full end-to-end model pipeline for Surgical Action Triplet Detection.
    """
    def __init__(self, use_refiner=True):
        super(TriDiffTransformer, self).__init__()
        
        # Hyperparameters (matched to project spec)
        d_model = 1024
        window_size = 8
        num_queries = 64
        
        # Components
        self.backbone = SwinBBackbone(pretrained=True, freeze_early_layers=True)
        self.t_encoder = BandedCausalTemporalEncoder(d_model=d_model)
        self.decoder = BipartiteBindingDecoder(d_model=d_model, num_queries=num_queries)
        
        self.use_refiner = use_refiner
        if self.use_refiner:
            self.refiner = DenoisingSemanticRefiner()

    def forward(self, x):
        """
        Args:
            x: (Batch, TemporalWindow, Channels, Height, Width)
        Returns:
            If refiner is OFF:
                logits_I, logits_V, logits_T, None, (z_I, z_V, z_T), spatial_feats
            If refiner is ON:
                logits_I, logits_V, logits_T, triplet_logits, (z_I, z_V, z_T), spatial_feats
        """
        # 1. Spatial Feature Extraction (Swin)
        spatial_feats = self.backbone(x) # (B, T, L_spatial, d_model)
        
        # 2. Causal Temporal Encoding
        # This gives us context-rich representation for each frame up to T
        temporal_feats = self.t_encoder(spatial_feats) # (B, T, d_model)
        
        # 3. Query Decoding & Bipartite Binding
        # Note: We take the last frame's representation (t=T-1) since we are 
        # predicting the triplet for the current frame, but we pass the whole 
        # sequence window as memory context for grounding.
        logits_I, logits_V, logits_T, latent_z = self.decoder(temporal_feats)
        
        # 4. Refinement (Phase 3 mostly)
        triplet_logits = None
        if self.use_refiner:
            logits_I, logits_V, logits_T, triplet_logits = self.refiner(logits_I, logits_V, logits_T)
            
        return logits_I, logits_V, logits_T, triplet_logits, latent_z, spatial_feats

if __name__ == "__main__":
    print("--- Testing Tri-Diff-Transformer Entire Pipeline ---")
    model = TriDiffTransformer(use_refiner=True).cuda()
    dummy_video_clip = torch.randn(2, 8, 3, 384, 384).cuda()
    
    out_I, out_V, out_T, out_Trip, z, s_f = model(dummy_video_clip)
    
    print("All forward passes successful without OOM or shape mismatch.")
    print(f"Final Triplet Logits Shape: {out_Trip.shape} (Expected B, 100)")
    print(f"Spatial Feats Shape: {s_f.shape}")
