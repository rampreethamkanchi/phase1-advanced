import torch
import torch.nn as nn
from .backbone import SwinBBackbone
from .t_encoder import BandedCausalTemporalEncoder
from .query_decoder import BipartiteBindingDecoder
from .refiner import DenoisingSemanticRefiner

class TemporalPhaseHead(nn.Module):
    """
    Plan A Innovation: 
    Leveraging the Triplet-aware Temporal features to predict surgical phases.
    We use a Temporal Transformer to capture long-range dependencies.
    """
    def __init__(self, d_model=1024, num_phases=7):
        super(TemporalPhaseHead, self).__init__()
        # We use a single layer Transformer Encoder to refine the temporal dynamics 
        # before classifying the phase. (Similar to MS-TCN++ approach conceptually but transformer-based).
        self.attn = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True)
        self.fc = nn.Linear(d_model, num_phases)
        
    def forward(self, z):
        # z: (B, T, d_model) from Temporal Encoder
        features = self.attn(z)
        # We classify the phase of the current frame (the last frame in the causal window T)
        last_frame_feat = features[:, -1, :] # (B, d_model)
        return self.fc(last_frame_feat)

class RelationalTransformer(nn.Module):
    """
    Plan A Innovation: Phase 3 Enhanced Surgical Scene Graph Generation.
    Uses Extracted Spatial Features to Enrich Node localized regions,
    and predict relationships (edges) + auxiliary attributes (active_energy).
    """
    def __init__(self, d_model=1024, num_edge_classes=18, num_node_classes=18, max_nodes=15):
        super().__init__()
        # +1 for padding if needed, but we clamp
        self.node_embed = nn.Embedding(num_node_classes + 1, d_model, padding_idx=-1)
        self.bbox_embed = nn.Linear(4, d_model)
        
        # Cross Attention between Vision Features and Nodes
        self.vision_to_node = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        
        # Edge Prediction Head (Subject-Object Pair)
        self.edge_predictor = nn.Sequential(
            nn.Linear(d_model * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_edge_classes)
        )
        
        # Attribute Enrichment: Energy State Classifier
        self.energy_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 1) # Binary Logit
        )

    def forward(self, visual_feats, nodes, bboxes):
        """
        visual_feats: (B, T, L_spatial, d_model)
        nodes: (B, MAX_NODES) -> class IDs
        bboxes: (B, MAX_NODES, 4) -> coords
        """
        B, T, L, D = visual_feats.shape 
        # For SG, we focus on the current/last frame
        v_f = visual_feats[:, -1, :, :] # (B, L, D)
        
        # Embed Nodes
        n_emb = self.node_embed(nodes.clamp(min=0)) # (B, N, D)
        b_emb = self.bbox_embed(bboxes) # (B, N, D)
        query = n_emb + b_emb 
        
        # Mask out padded nodes (where node_id == -1)
        # nodes is (B, N). Mask is True where we want to IGNORE (padding)
        key_padding_mask = (nodes == -1) # (B, N)
        
        # Enrich nodes with visual context from Swin Backbone
        # vision_to_node(query, key, value)
        # Note: We don't strictly *need* to mask out the queries if they don't affect other nodes,
        # but to be clean, we can.
        enriched_nodes, _ = self.vision_to_node(
            query=query, 
            key=v_f, 
            value=v_f,
            # query mask has to be applied differently, but typically we just pad
        ) # (B, N, D)
        
        # Mask out padded nodes so they don't pollute global energy
        N = enriched_nodes.shape[1]
        mask_expanded = key_padding_mask.unsqueeze(-1).expand(B, N, D)
        enriched_nodes_clean = enriched_nodes.masked_fill(mask_expanded, 0.0)
        
        # Predict Edges
        # Create pairwise combinations [N_subj, N_obj]
        x_i = enriched_nodes.unsqueeze(2).expand(B, N, N, D)
        x_j = enriched_nodes.unsqueeze(1).expand(B, N, N, D)
        pair_feats = torch.cat([x_i, x_j], dim=-1) # (B, N, N, 2D)
        
        # (B, N, N, num_edge_classes)
        edge_logits = self.edge_predictor(pair_feats) 
        
        # Energy Attr: Global pool of valid enriched nodes
        # Avoid zero-division
        valid_counts = (~key_padding_mask).sum(dim=1, keepdim=True).clamp(min=1)
        global_node_feat = enriched_nodes_clean.sum(dim=1) / valid_counts # (B, D)
        
        energy_logits = self.energy_head(global_node_feat).squeeze(-1) # (B)
        
        return edge_logits, energy_logits

class TriDiffTransformer(nn.Module):
    """
    Tri-Diff-Transformer (TDT).
    Full end-to-end model pipeline for Surgical Action Triplet Detection & Phase Detection.
    """
    def __init__(self, use_refiner=True, plan='A', num_phases=7):
        super(TriDiffTransformer, self).__init__()
        
        # Hyperparameters (matched to project spec)
        d_model = 1024
        window_size = 8
        num_queries = 64
        
        self.plan = plan

        # Components
        self.backbone = SwinBBackbone(pretrained=True, freeze_early_layers=True)
        
        # If Plan B, we might freeze the backbone completely and rely on a pre-trained SOTA model's config,
        # but for simplicity in architecture orchestration, we keep it unified here and handle freezing in train.py
        
        self.t_encoder = BandedCausalTemporalEncoder(d_model=d_model)
        self.decoder = BipartiteBindingDecoder(d_model=d_model, num_queries=num_queries)
        
        self.use_refiner = use_refiner
        if self.use_refiner:
            self.refiner = DenoisingSemanticRefiner()

        # Phase 2: Phase Detection Head
        if self.plan == 'A':
            self.phase_head = TemporalPhaseHead(d_model=d_model, num_phases=num_phases)
        elif self.plan == 'B':
            # Plan B: Integrative SOTA Path. 
            self.phase_head = nn.Sequential(
                nn.Linear(d_model, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_phases)
            )
            
        # Phase 3: Enhanced Scene Graph Generator
        if self.plan == 'A':
            self.relational_transformer = RelationalTransformer(d_model=d_model)

    def forward(self, x, nodes=None, bboxes=None):
        """
        Args:
            x: (Batch, TemporalWindow, Channels, Height, Width)
            nodes: Optional (B, MAX_NODES) for Phase 3
            bboxes: Optional (B, MAX_NODES, 4) for Phase 3
        Returns:
            logits_I, logits_V, logits_T, triplet_logits, latent_z, spatial_feats, phase_logits, edge_logits, energy_logits
        """
        # 1. Spatial Feature Extraction (Swin)
        spatial_feats = self.backbone(x) # (B, T, L_spatial, d_model)
        
        # 2. Causal Temporal Encoding
        # This gives us context-rich representation for each frame up to T
        temporal_feats = self.t_encoder(spatial_feats) # (B, T, d_model)
        
        # Phase 2 Detection (Branching off temporal features)
        if self.plan == 'A':
            phase_logits = self.phase_head(temporal_feats)
        else:
            last_frame_temp = temporal_feats[:, -1, :]
            phase_logits = self.phase_head(last_frame_temp)

        # 3. Query Decoding & Bipartite Binding
        logits_I, logits_V, logits_T, latent_z = self.decoder(temporal_feats)
        
        # 4. Refinement 
        triplet_logits = None
        if self.use_refiner:
            logits_I, logits_V, logits_T, triplet_logits = self.refiner(logits_I, logits_V, logits_T)
            
        # Phase 3: Scene Graph Generation (If nodes are provided)
        edge_logits, energy_logits = None, None
        if nodes is not None and bboxes is not None and self.plan == 'A':
            edge_logits, energy_logits = self.relational_transformer(spatial_feats, nodes, bboxes)
            
        return logits_I, logits_V, logits_T, triplet_logits, latent_z, spatial_feats, phase_logits, edge_logits, energy_logits

if __name__ == "__main__":
    print("--- Testing Tri-Diff-Transformer Entire Pipeline ---")
    model = TriDiffTransformer(use_refiner=True, plan='A').cuda()
    dummy_video_clip = torch.randn(2, 8, 3, 384, 384).cuda()
    nodes = torch.randint(0, 18, (2, 15)).cuda()
    bboxes = torch.randn(2, 15, 4).cuda()
    
    out_I, out_V, out_T, out_Trip, z, s_f, phase, edges, energy = model(dummy_video_clip, nodes, bboxes)
    
    print("All forward passes successful without OOM or shape mismatch.")
    print(f"Final Triplet Logits Shape: {out_Trip.shape} (Expected B, 100)")
    print(f"Spatial Feats Shape: {s_f.shape}")
    print(f"Phase Logits Shape: {phase.shape} (Expected B, 7)")
    print(f"Edge Logits Shape: {edges.shape} (Expected B, N, N, 18)")
    print(f"Energy Logits Shape: {energy.shape} (Expected B)")
