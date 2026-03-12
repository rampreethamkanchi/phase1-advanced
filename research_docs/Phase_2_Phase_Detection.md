# Methodology Context: Phase 2 Phase Detection

This document contains the complete source code for all modules contributing to this phase. Use this as context for writing the methodology section of the research paper.

## File: `src/dataset_cholecT50.py`

```python
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pickle

class CholecT50Dataset(Dataset):
    """
    Dataset for CholecT50 which includes both Triplets and Surgical Phase annotations.
    """
    def __init__(self, dataset_dir, split='train', cache_dir="./data/cache", transform=None, window_size=8):
        self.dataset_dir = dataset_dir
        self.cache_dir = cache_dir
        self.transform = transform
        self.window_size = window_size
        self.split = split
        
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_path = os.path.join(self.cache_dir, f"cholect50_parsed_{split}.pkl")
        
        self.samples = self._load_or_build_index()

    def _get_split_videos(self):
        # Total 50 videos. We use first 35 for train, 5 for val, 10 for test, similar to standard splits.
        # Or standard CholecT50 split: train: 35, val: 5, test: 10
        # Let's get all video IDs
        all_videos = sorted([d.split('.')[0] for d in os.listdir(os.path.join(self.dataset_dir, "labels")) if d.endswith('.json')])
        if self.split == 'train':
            return all_videos[:35]
        elif self.split == 'val':
            return all_videos[35:40]
        elif self.split == 'test':
            return all_videos[40:]
        else:
            raise ValueError("Split must be train, val, or test")

    def _load_or_build_index(self):
        if os.path.exists(self.cache_path):
            print(f">>> [Dataset] Loading cached CholecT50 index from {self.cache_path}...")
            with open(self.cache_path, "rb") as f:
                samples = pickle.load(f)
            return samples
        
        print(f">>> [Dataset] Building CholecT50 index for {self.split} split...")
        split_vids = self._get_split_videos()
        samples = []
        
        for vid in split_vids:
            print(f"    -> Parsing {vid}...")
            img_dir = os.path.join(self.dataset_dir, "videos", vid)
            label_file = os.path.join(self.dataset_dir, "labels", f"{vid}.json")
            
            if not os.path.exists(img_dir) or not os.path.exists(label_file):
                print(f"Warning: Missing data for {vid}")
                continue
                
            with open(label_file, "r") as f:
                d = json.load(f)
                annotations = d.get("annotations", d)
                
            frame_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
            
            v_start = len(samples)
            for i, frame_file in enumerate(frame_files):
                frame_idx_str = str(i)
                if frame_idx_str not in annotations:
                    continue
                    
                frame_anns = annotations[frame_idx_str]
                
                # Parse annotations
                phase_id = int(frame_anns[0][-1]) # Phase is the 15th element (index 14)
                
                # Multi-hot encoded triplets (100 classes as per label mapping)
                triplet_multihot = [0.0] * 100
                for ann in frame_anns:
                    triplet_id = int(ann[0])
                    if triplet_id != -1:
                        triplet_multihot[triplet_id] = 1.0
                
                samples.append({
                    'img_path': os.path.join(img_dir, frame_file),
                    'video_id': vid,
                    'frame_id': i,
                    'video_start_idx': v_start,
                    'phase': phase_id,
                    'triplet': triplet_multihot
                })
                
        print(f">>> [Dataset] Saving parsed cache to {self.cache_path}...")
        with open(self.cache_path, "wb") as f:
            pickle.dump(samples, f)
            
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        current_sample = self.samples[idx]
        video_start_idx = current_sample['video_start_idx']
        
        frames = []
        indices = [max(idx - i, video_start_idx) for i in range(self.window_size - 1, -1, -1)]
        
        for target_idx in indices:
            frame_path = self.samples[target_idx]['img_path']
            img = Image.open(frame_path).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            frames.append(img)
            
        frames_tensor = torch.stack(frames, dim=0)
        phase_gt = torch.tensor(current_sample['phase'], dtype=torch.long)
        triplet_gt = torch.tensor(current_sample['triplet'], dtype=torch.float32)
        
        return frames_tensor, (phase_gt, triplet_gt, current_sample['video_id'])

def build_transforms(is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop((448, 448), scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def get_dataloader_t50(dataset_dir, split='train', batch_size=4, num_workers=16, window_size=8, pin_memory=False):
    transform = build_transforms(is_train=(split == 'train'))
    dataset = CholecT50Dataset(dataset_dir, split=split, transform=transform, window_size=window_size)
    
    dl = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=(split == 'train'), 
        num_workers=num_workers,
        pin_memory=pin_memory 
    )
    
    return dl, dataset

if __name__ == "__main__":
    print("--- Testing Dataset & Caching (CholecT50) ---")
    TEST_DATASET_DIR = "/raid/manoranjan/rampreetham/CholecT50"
    
    dl, ds = get_dataloader_t50(TEST_DATASET_DIR, split='val', batch_size=2, num_workers=4)
    
    for batch_idx, (frames, labels) in enumerate(dl):
        phase, trip, vid = labels
        print(f"Batch {batch_idx}:")
        print(f" - Frames Shape: {frames.shape} (Expected: B, T, C, H, W)")
        print(f" - Phase GT Shape: {phase.shape}")
        print(f" - Triplet GT Shape: {trip.shape}")
        break 

```

---

## File: `src/models/t_encoder.py`

```python
import torch
import torch.nn as nn

class CausalPositionalEncoding(nn.Module):
    """
    Standard 1D positional encoding to inject temporal ordering information 
    into the sequence of frames before processing them. 
    """
    def __init__(self, d_model, max_len=100):
        super(CausalPositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: (B, T, C)
        Returns: x + pe
        """
        B, T, C = x.shape
        return x + self.pe[:T, :].unsqueeze(0)


class BandedCausalTemporalEncoder(nn.Module):
    """
    Banded Causal Temporal Encoder.
    Operates on the spatial-pooled features generated by the Backbone.
    
    Instead of LSTMs, this uses a Transformer Encoder with a strict *causal mask*. 
    A causal mask ensures that processing for frame `t` ONLY attends to `t` and 
    previous frames, strictly preventing data leakage from the future in a surgical setting.
    """
    def __init__(self, d_model=1024, nhead=8, num_layers=3, dim_feedforward=2048, dropout=0.1):
        super(BandedCausalTemporalEncoder, self).__init__()
        
        self.pos_encoder = CausalPositionalEncoding(d_model=d_model)
        
        # standard transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True # We will pass (B, T, C) into the transformer
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def generate_causal_mask(self, sz):
        """
        Generates an upper-triangular matrix of -inf, with zeros on diag/lower triangle.
        This forces the self-attention to only look back in time.
        """
        # torch.triu generates upper triangular. True where row < col (i.e. future)
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x):
        """
        Args:
            x: (B, T, NumSpatialTokens, Channels). 
               e.g., from Swin-B: (2, 8, 144, 1024)
        Returns:
            encoded_frames: (B, T, d_model). Summarized temporal state per frame.
        """
        B, T, L, C = x.shape
        
        # For the temporal encoder, we want to summarize the spatial information first 
        # before running it through time. We can achieve this via average pooling.
        # So we pool the 144 tokens down to 1 feature vector per frame.
        # x_pooled: (B, T, C)
        x_pooled = x.mean(dim=2) 
        
        # Inject temporal position info
        x_pos = self.pos_encoder(x_pooled)
        
        # Create causal mask (T x T)
        causal_mask = self.generate_causal_mask(T).to(x.device)
        
        # Transformer pass
        # The transformer output naturally matches the causal masking, meaning 
        # out[:, t, :] represents the historical context up to frame t.
        out = self.transformer_encoder(x_pos, mask=causal_mask) # (B, T, C)
        
        return out

if __name__ == "__main__":
    print("--- Testing Banded Causal Temporal Encoder ---")
    model = BandedCausalTemporalEncoder().cuda()
    
    # Dummy features representing Swin-B output (B=2, T=8, Spatial=144, C=1024)
    dummy_feats = torch.randn(2, 8, 144, 1024).cuda()
    
    encoded = model(dummy_feats)
    print(f"Input Feature Shape: {dummy_feats.shape}")
    print(f"Temporal Encoded Shape: {encoded.shape} (Expected: B, T, d_model = 2, 8, 1024)")
    print("--- Test Passed ---")

```

---

## File: `src/losses/supcon.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MemoryBank(nn.Module):
    """
    FIFO Queue representing highly discriminative tail classes.
    Stores latent embedding 'z' for successfully classified rare classes.
    """
    def __init__(self, feature_dim=1024, queue_size=256, tail_classes=None):
        super(MemoryBank, self).__init__()
        self.feature_dim = feature_dim
        self.queue_size = queue_size
        
        # For simplicity, if tail_classes is None, this serves as a general 
        # class-wise memory bank. If provided, we only store these classes.
        self.tail_classes = tail_classes if tail_classes is not None else []
        self.num_classes = len(self.tail_classes) if tail_classes else 100 
        
        # Register buffers so they are saved with checkpoints but don't require gradients
        self.register_buffer("queue", torch.randn(self.num_classes, queue_size, feature_dim))
        self.register_buffer("queue_ptr", torch.zeros(self.num_classes, dtype=torch.long))
        
        # Normalize the random initialization
        self.queue = F.normalize(self.queue, dim=-1)

    @torch.no_grad()
    def update(self, embeddings, labels):
        """
        Args:
            embeddings: (Batch, d) latent queries from the decoder
            labels: (Batch,) argmax classification result (or ground truth during training)
        """
        embeddings = F.normalize(embeddings.float(), dim=-1)
        
        for b in range(embeddings.shape[0]):
            lbl_idx = labels[b].item()
            
            # Use continuous mapped indexing if tail classes provided, otherwise direct global idx
            store_idx = -1
            if self.tail_classes:
                if lbl_idx in self.tail_classes:
                    store_idx = self.tail_classes.index(lbl_idx)
            else:
                store_idx = lbl_idx
                
            if store_idx != -1:
                ptr = int(self.queue_ptr[store_idx])
                
                # Replace the oldest embedding with the new one
                self.queue[store_idx, ptr] = embeddings[b]
                
                # Increment and wrap
                ptr = (ptr + 1) % self.queue_size
                self.queue_ptr[store_idx] = ptr

    def get_positives(self, class_idx, num_samples=10):
        """ Retrieves samples from the memory bank to serve as contrastive positives. """
        store_idx = -1
        if self.tail_classes:
             if class_idx in self.tail_classes:
                 store_idx = self.tail_classes.index(class_idx)
        else:
             store_idx = class_idx
             
        if store_idx != -1:
             # Just return a random subset or the top N
             # Here we return the first `num_samples` for simplicity, could add random permutation
             # We MUST clone to detach the view from the inplace-updated queue buffer!
             return self.queue[store_idx, :num_samples, :].clone()
        return None

class TailBoostedSupConLoss(nn.Module):
    """
    Supervised Contrastive Loss interfacing with a Tail-Class Memory Bank.
    
    Pulls tail-class embeddings close to successful historical tail-class queries,
    and pushes them away from head-class embeddings.
    """
    def __init__(self, temperature=0.07, feature_dim=1024, queue_size=512, tail_classes=None):
        super(TailBoostedSupConLoss, self).__init__()
        self.temperature = temperature
        self.memory = MemoryBank(feature_dim, queue_size, tail_classes)
        
    def forward(self, embeddings, labels):
        """
        Note: SupCon requires non-multi-label formulation usually, or a modified 
        multi-label approach. For CholecT45 Triplets, we treat the 100 triplets 
        as independent IDs for this loss.
        """
        device = embeddings.device
        
        # Disable autocast, keep everything in FP32 to prevent exponent overflow
        with torch.amp.autocast('cuda', enabled=False):
            # Cast to FP32 to prevent exponent overflow (exp(14.28) > FP16 max)
            embeddings = embeddings.float()
            embeddings = F.normalize(embeddings, dim=-1)
            
            loss = torch.tensor(0.0, device=device)
            count = 0
            
            # A very simplistic contrastive iteration
            # In a real batched implementation, you process the sim matrix.
            for b in range(embeddings.shape[0]):
                lbl = labels[b] # Assuming single categorical triplet label for simplicity of contrastive
                
                # If multi-label, we take the hardest (rarest) positive label
                # But assuming lbl is a scalar ID here:
                
                positives = self.memory.get_positives(lbl.item())
                if positives is not None:
                    positives = positives.to(device).float()
                    num_pos = positives.shape[0]
                    
                    # Compare embedding to positives
                    # (1, d) @ (d, N) -> (1, N)
                    sim_pos = torch.matmul(embeddings[b].unsqueeze(0), positives.transpose(0, 1)) / self.temperature
                    
                    # For negatives, we compare against everything else in the batch
                    mask = (labels != lbl)
                    negatives = embeddings[mask]
                    
                    if negatives.shape[0] > 0:
                         sim_neg = torch.matmul(embeddings[b].unsqueeze(0), negatives.transpose(0, 1)) / self.temperature
                         
                         # Stable InfoNCE formulation using logsumexp to avoid NaN and overflow
                         log_num = torch.logsumexp(sim_pos, dim=1)
                         sim_all = torch.cat([sim_pos, sim_neg], dim=1)
                         log_denom = torch.logsumexp(sim_all, dim=1)
                         
                         loss = loss + (log_denom - log_num).squeeze()
                         count += 1
                         
            if count > 0:
                 loss = loss / count
                 
            # Important: Update the memory bank with non-differentiable clones!
            self.memory.update(embeddings.detach(), labels)
            
        return loss

if __name__ == "__main__":
    print("--- Testing SupCon Loss ---")
    # T45 has 100 triplet classes. We assign a few arbitrary tail classes
    tail_classes = [90, 91, 92, 93, 94] 
    loss_fn = TailBoostedSupConLoss(tail_classes=tail_classes).cuda()
    
    # B=4
    dummy_z = torch.randn(4, 1024).cuda()
    dummy_labels = torch.tensor([90, 5, 90, 2]).cuda()
    
    loss = loss_fn(dummy_z, dummy_labels)
    print(f"SupCon Loss output: {loss.item()}")
    
    print("--- Test Passed ---")

```

---

## File: `src/models/tdt.py`

```python
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

```

---

## File: `src/train.py`

```python
import os
import time
import argparse
import datetime
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
torch.autograd.set_detect_anomaly(True)
import torch.optim as optim

# Import Custom Modules
from dataset import get_dataloader
from dataset_cholecT50 import get_dataloader_t50
from models.tdt import TriDiffTransformer
from losses.asl import AsymmetricLossOptimized
from losses.mcl import MutualChannelLoss
from losses.supcon import TailBoostedSupConLoss
from eval import evaluate_model

def setup_logger(log_dir="./logs"):
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"run_{ts}.log")
    
    # Configure logging to go to both file and console
    # This satisfies rule 0: Dual-Stream Output & Insightful Observations
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()
    
    import sys
    import warnings
    
    # Redirect warnings to the logging system
    logging.captureWarnings(True)
    
    # Capture unhandled exceptions
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.error("Uncaught exception occurred:", exc_info=(exc_type, exc_value, exc_traceback))
        
    sys.excepthook = handle_exception
    
    return logger, log_file

def calculate_phase(epoch, total_epochs=80):
    """ 
    Determines the current training phase (Curriculum Learning).
    
    Why Phased Training?
    - Phase 1 (0-10): Warmup. We focus on basic instrument/verb localization. 
      The model learns 'where' and 'what' without the complex refiner.
    - Phase 2 (10-25): Tail Boost. We activate the Contrastive Memory Bank. 
      This helps the model 'memorize' rare triplets (tail classes).
    - Phase 3 (25+): Refinement. The Denoising Semantic Refiner (DSR) is engaged.
      The model now uses clinical logic to fix impossible predictions.
    """
    if epoch < 10:
        return 1 
    elif epoch < 25:
        return 2 
    else:
        return 3 

def train_one_epoch(epoch, model, dataloader, optimizer, criterion_dict, device, phase, logger, scaler, scheduler, args):
    model.train()
    
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} Phase {phase}")
    
    optimizer.zero_grad()
    
    for batch_idx, (frames, labels) in enumerate(progress_bar):
        # Optimization: non_blocking=True for faster GPU transfer
        frames = frames.to(device, non_blocking=True)
        
        # Check if doing Phase 2 (CholecT50 with phases) or Phase 1 (CholecT45)
        is_phase2_task = args.dataset_type == 'cholecT50'
        
        if is_phase2_task:
            phase_gt, triplet_gt, _ = labels
            phase_gt = phase_gt.to(device, non_blocking=True)
            triplet_gt = triplet_gt.to(device, non_blocking=True)
            # Phase 2 model doesn't use the intermediate IVT ground truths in its main loop if we just want triplet and phase
        else:
            inst_gt, verb_gt, target_gt, triplet_gt, _ = labels
            inst_gt = inst_gt.to(device, non_blocking=True)
            verb_gt = verb_gt.to(device, non_blocking=True)
            target_gt = target_gt.to(device, non_blocking=True)
            triplet_gt = triplet_gt.to(device, non_blocking=True)
        
        # Turn off refiner gradients / usage if during early phases
        use_refiner = (phase == 3)
        model.use_refiner = use_refiner
        
        # Mixed Precision Autocast (Standardized for        # Mixed Precision
        with torch.amp.autocast('cuda', enabled=args.use_amp):
            # TDT returns 9 values now (added edge_logits, energy_logits for Phase 3)
            logits_I, logits_V, logits_T, triplet_logits, (z_I, z_V, z_T), spatial_feats, phase_logits, _, _ = model(frames)
            
            loss = 0.0
            if not is_phase2_task:
                # 1. Base Multi-Label Detection Loss (Phase 1)
                loss_I = criterion_dict['asl'](logits_I, inst_gt)
                loss_V = criterion_dict['asl'](logits_V, verb_gt)
                loss_T = criterion_dict['asl'](logits_T, target_gt)
                loss = loss + loss_I + loss_V + loss_T
            
            if is_phase2_task:
                # Phase Detection Loss (CrossEntropy)
                # Ensure phase_logits matches shape of phase_gt: phase_logits is (B, 7)
                loss_phase = criterion_dict['ce'](phase_logits, phase_gt)
                loss = loss + (args.lam_phase * loss_phase)
                
            # 2. Mutual Channel Loss (Always Active)
            if args.lam_mcl > 0:
                loss_mcl = criterion_dict['mcl'](spatial_feats)
                loss = loss + loss_mcl
            
            # 3. SupCon Loss (Phase 2+) (Assuming triplet features z_V holds for action representation)
            if phase >= 2 and args.lam_supcon > 0:
                 triplet_class = torch.argmax(triplet_gt, dim=1)
                 loss_supcon = criterion_dict['supcon'](z_V, triplet_class) 
                 loss = loss + (args.lam_supcon * loss_supcon)
                 
            # 4. Refiner KL/ASL Loss (Phase 3)
            # if is_phase2_task, the refiner still outputs triplet probabilities
            if use_refiner and triplet_logits is not None:
                 loss_refiner = criterion_dict['asl'](triplet_logits, triplet_gt)
                 loss = loss + (args.lam_dsr * loss_refiner)
                 
            loss = loss / args.grad_accum_steps
             
        # Scale and backward step
        scaler.scale(loss).backward()
        
        if ((batch_idx + 1) % args.grad_accum_steps == 0) or (batch_idx + 1 == len(dataloader)):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        current_batch_loss = loss.item() * args.grad_accum_steps
        total_loss += current_batch_loss
        progress_bar.set_postfix({'loss': f"{current_batch_loss:.4f}"})
        
        log_freq = 5 if args.sample_run else 50
        if batch_idx % log_freq == 0:
             avg_loss_so_far = total_loss / (batch_idx + 1)
             msg = f"[Epoch {epoch} | Batch {batch_idx}/{len(dataloader)}] Loss: {current_batch_loss:.4f} (Avg: {avg_loss_so_far:.4f}) | Phase {phase}"
             if is_phase2_task:
                 msg += f" -> [Learning phase features ({args.plan})]"
             logger.info(msg)
                         
        if args.sample_run and batch_idx >= 5:
             logger.info("Sample run: Breaking batch loop early for quick test.")
             break
                         
    avg_loss = total_loss / (batch_idx + 1)
    logger.info(f"*** Epoch {epoch} completed. Average Loss: {avg_loss:.4f} ***")
    return avg_loss

def main(args):
    logger, log_file = setup_logger()
    logger.info(f"Starting Training Process. Log file saved to: {log_file}")
    logger.info(f"Arguments: {args}")
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    logger.info(f"Utilizing Device: {device}")
    
    # 1. Dataset Initialization
    logger.info(f"Initializing DataLoaders for {args.dataset_type} from: {args.dataset_dir}")
    if args.dataset_type == 'cholecT50':
        train_dl, _ = get_dataloader_t50(args.dataset_dir, split='train', batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_memory)
        val_dl, _ = get_dataloader_t50(args.dataset_dir, split='val', batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_memory)
    else:
        train_dl, _ = get_dataloader(args.dataset_dir, split='train', batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_memory)
        val_dl, _ = get_dataloader(args.dataset_dir, split='val', batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_memory)
    
    # 2. Model Initialization
    logger.info(f"Initializing Tri-Diff-Transformer (TDT) Backbone (Plan {args.plan})...")
    model = TriDiffTransformer(use_refiner=True, plan=args.plan, num_phases=7)
    model = model.to(device)
    
    # 3. Optimizers & Losses
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.t_encoder.parameters()) + list(model.decoder.parameters())
    if args.dataset_type == 'cholecT50':
        head_params += list(model.phase_head.parameters())
        
    if model.use_refiner:
        head_params += list(model.refiner.parameters())
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': args.lr_backbone},
        {'params': head_params, 'lr': args.lr_heads}
    ], weight_decay=args.weight_decay)
    
    # Schedulers
    warmup_iters = args.warmup_epochs
    main_iters = args.epochs - warmup_iters
    
    scheduler_warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_iters)
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=main_iters, eta_min=1e-7)
    
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_iters]
    )
    
    # Mixed Precision Scaler
    scaler = torch.amp.GradScaler('cuda', enabled=args.use_amp)
    
    # Load Tail Classes from precomputed stats
    import json
    stats_path = "data/cache/stats.json"
    tail_classes = list(range(85, 100)) # Fallback
    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            stats = json.load(f)
        tail_classes = stats.get('tail_classes', tail_classes)
        logger.info(f"Loaded {len(tail_classes)} tail classes for SupCon from precomputed stats.")

    criterion_dict = {
        'asl': AsymmetricLossOptimized().to(device),
        'mcl': MutualChannelLoss(alpha=args.lam_mcl).to(device),
        'supcon': TailBoostedSupConLoss(
             temperature=args.supcon_temp, 
             feature_dim=1024, 
             queue_size=args.supcon_bank_size, 
             tail_classes=tail_classes
        ).to(device),
        'ce': nn.CrossEntropyLoss().to(device)
    }
    
    # 4. Master Training Loop
    num_epochs = args.epochs
    best_mAP = -1.0  # Initialize with a negative value to ensure the first eval is recorded
    for epoch in range(1, num_epochs + 1):
        if args.sample_run:
            phase = min(epoch, 3) 
        else:
            phase = calculate_phase(epoch, total_epochs=args.epochs)
            
        logger.info(f"--- Launching Epoch {epoch}/{num_epochs} (Phase {phase}) ---")
        
        # Train
        train_loss = train_one_epoch(
            epoch, model, train_dl, optimizer, criterion_dict, device, phase, logger, scaler, scheduler, args
        )
        
        # Step the epoch-level scheduler
        scheduler.step()
        logger.info(f"Current LR (Backbone): {optimizer.param_groups[0]['lr']:.7f} | LR (Heads): {optimizer.param_groups[1]['lr']:.7f}")
        
        # Validation Pass on Val Set (Much faster than full dataset)
        logger.info(f"--- Running Evaluation pass for Epoch {epoch} ---")
        eval_metric = evaluate_model(model, val_dl, device, logger, phase, is_sample_run=args.sample_run, is_phase2_task=(args.dataset_type == 'cholecT50'))
        
        # Sample mode short-circuit
        if args.sample_run and epoch == 3:
            logger.info("Sample run complete. Terminating.")
            break
        
        # Track Best Performance
        is_best = eval_metric > best_mAP
        if is_best:
            best_mAP = eval_metric
            logger.info(f"New Best Target Metric: {best_mAP:.4f} attained at Epoch {epoch}")
            # We also save a 'best_model.pt' to always have the absolute top performer available,
            # regardless of the 5-epoch interval. This ensures no peak performance is lost.
            # Specific naming to prevent overwriting different tasks
            best_path = f"logs/best_{args.dataset_type}_{args.plan}_phase{phase}.pt"
            torch.save(model.state_dict(), best_path)
            logger.info(f"Absolute best model updated at: {best_path}")

        # Save Checkpoint every 5 epochs (or last epoch) ONLY if it's the best so far
        # This satisfies the user's request: "save... after every 5 epochs... only if its better than earlier"
        if epoch % 5 == 0 or epoch == num_epochs:
             if is_best:
                 checkpoint_path = f"logs/checkpoint_{args.dataset_type}_{args.plan}_ep{epoch}.pt"
                 torch.save(model.state_dict(), checkpoint_path)
                 logger.info(f"Interval Checkpoint: Performance improved! Saved to: {checkpoint_path}")
             else:
                 logger.info(f"Interval Checkpoint: Epoch {epoch} performance ({eval_metric:.4f}) "
                             f"did not exceed earlier best ({best_mAP:.4f}). Skipping save.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Tri-Diff-Transformer Training Script")
    parser.add_argument("--dataset_dir", type=str, default="/raid/manoranjan/rampreetham/CholecT50")
    parser.add_argument("--dataset_type", type=str, default="cholecT50", choices=['cholecT45', 'cholecT50'])
    parser.add_argument("--plan", type=str, default="A", choices=['A', 'B'], help="Architectural path for phase 2. A: Temporal Transformer, B: Mocked TeCNO light MLP")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch Size. Warning: 64 requires high VRAM. Dial down to 32 and use grad_accum=4 if OOM occurs.")
    parser.add_argument("--grad_accum_steps", type=int, default=2, help="Gradient Accumulation Steps")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--lr_heads", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    
    # Advanced / Specific Configurations
    parser.add_argument("--use_amp", action="store_true", default=True, help="Use Automatic Mixed Precision")
    parser.add_argument("--lam_mcl", type=float, default=1.0)
    parser.add_argument("--lam_supcon", type=float, default=0.5)
    parser.add_argument("--lam_dsr", type=float, default=0.1)
    parser.add_argument("--lam_phase", type=float, default=1.0)
    parser.add_argument("--supcon_temp", type=float, default=0.07)
    parser.add_argument("--supcon_bank_size", type=int, default=512)
    parser.add_argument("--pin_memory", action="store_true", default=False, help="Enable pin_memory in DataLoader (Warning: can cause 'invalid argument' error on some DGX setups)")
    
    parser.add_argument("--sample_run", action="store_true", help="Run a quick 2-epoch test")
    
    args = parser.parse_args()
    main(args)

```

---

## File: `src/eval.py`

```python
import os
import torch
import numpy as np
from tqdm import tqdm
from ivtmetrics import Recognition

from sklearn.metrics import average_precision_score
import json

def evaluate_model(model, dataloader, device, logger, phase=3, is_sample_run=False, is_phase2_task=False):
    """
    Evaluates the model on the provided dataloader using standard video-wise mAP metrics.
    Uses ivtmetrics to compute mAPs for Instruments, Verbs, Targets, and Triplets.
    """
    model.eval()
    
    # Initialize ivtmetrics recognition evaluator for 100 triplets
    evaluator = Recognition(num_class=100) 
    
    # Load Triplet mapping for Bayesian product approximation (Compute once, reuse)
    id_to_ivt = None
    stats_path = "data/cache/stats.json"
    if os.path.exists(stats_path):
        try:
            with open(stats_path, "r") as f:
                stats = json.load(f)
            # Map keys in JSON are strings, convert back to int
            id_to_ivt = {int(k): v for k, v in stats['id_to_ivt'].items()}
        except Exception:
            pass

    logger.info(f"Starting Evaluation Pass (Phase {phase})...")
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating")
        current_video = None
        
        phase_correct = 0
        phase_total = 0
        
        for batch_idx, (frames, labels) in enumerate(progress_bar):
            if batch_idx % 10 == 0:
                logger.info(f"Evaluating Batch {batch_idx}/{len(dataloader)}")
                
            frames = frames.to(device)
            # Check if doing Phase 2 (CholecT50 with phases) or Phase 1 (CholecT45)
            if is_phase2_task:
                phase_gt = labels[0].to(device)
                triplet_gt = labels[1].cpu().numpy()
                video_ids = labels[2]
            else:
                triplet_gt = labels[3].cpu().numpy()
                video_ids = labels[4]
            
            # Forward pass
            model.use_refiner = (phase == 3)
            # TDT returns 9 values now
            logits_I, logits_V, logits_T, triplet_logits, _, _, phase_logits, _, _ = model(frames)
            
            if is_phase2_task:
                 _, predicted_phases = torch.max(phase_logits, 1)
                 phase_total += phase_gt.size(0)
                 phase_correct += (predicted_phases == phase_gt).sum().item()
            
            # 1. Base Multi-Label Detection
            preds_I = torch.sigmoid(logits_I).cpu().numpy()
            preds_V = torch.sigmoid(logits_V).cpu().numpy()
            preds_T = torch.sigmoid(logits_T).cpu().numpy()
            
            # 2. Triplet Prediction Logic
            if phase == 3 and triplet_logits is not None:
                # Use refined triplet logits directly
                preds_trip = torch.sigmoid(triplet_logits).cpu().numpy()
            elif id_to_ivt is not None:
                # Bayesian Product Approximation: P(I)*P(V)*P(T)
                # Build (B, 100) matrix from component preds
                B = preds_I.shape[0]
                preds_trip = np.zeros((B, 100), dtype=np.float32)
                for tid, (i, v, t) in id_to_ivt.items():
                    # Probability of triplet k is product of its parts
                    preds_trip[:, tid] = preds_I[:, i] * preds_V[:, v] * preds_T[:, t]
            else:
                # Fallback if no map
                preds_trip = np.zeros_like(triplet_gt)
                
            # Update evaluator per item to beautifully handle video boundaries
            for i in range(len(triplet_gt)):
                vid = video_ids[i]
                if current_video is None:
                    current_video = vid
                elif current_video != vid:
                    evaluator.video_end()
                    current_video = vid
                
                # update single frame
                evaluator.update(triplet_gt[i:i+1], preds_trip[i:i+1])
            
            if is_sample_run and batch_idx >= 5:
                 logger.info("Sample run: Breaking eval loop early for quick test.")
                 break
            
        # Final video end
        if current_video is not None:
            evaluator.video_end()
            
    # Compute mAPs using ivtmetrics
    # We compute video-wise AP (Recommended by benchmark)
    
    phase_acc = (phase_correct / phase_total) if phase_total > 0 else 0.0
    
    def get_metric(comp):
        try:
            res = evaluator.compute_video_AP(comp)
            return res.get("mAP", 0.0)
        except:
            return 0.0

    mAP_I = get_metric('i')
    mAP_V = get_metric('v')
    mAP_T = get_metric('t')
    mAP_IVT = get_metric('ivt')

    logger.info(f"--- Evaluation Results (Phase {phase}) ---")
    if is_phase2_task:
        logger.info(f"Triplet mAP:    {mAP_IVT:.4f}")
        logger.info(f"Phase Accuracy: {phase_acc:.4f}")
        return mAP_IVT + phase_acc # Return a combined metric for 'best model' tracking during Phase 2
    else:
        logger.info(f"Instrument mAP: {mAP_I:.4f}")
        logger.info(f"Verb mAP:       {mAP_V:.4f}")
        logger.info(f"Target mAP:     {mAP_T:.4f}")
        logger.info(f"Triplet mAP:    {mAP_IVT:.4f}{' (Bayesian Proxy)' if phase < 3 else ''}")
        return mAP_IVT

if __name__ == "__main__":
    print("--- Eval module structure OK ---")

```

---

