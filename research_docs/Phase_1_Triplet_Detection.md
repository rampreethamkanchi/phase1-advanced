# Methodology Context: Phase 1 Triplet Detection

This document contains the complete source code for all modules contributing to this phase. Use this as context for writing the methodology section of the research paper.

## File: `src/dataset.py`

```python
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import pickle

class CholecT45Dataset(Dataset):
    """
    Custom PyTorch Dataset for CholecT45 Action Triplet Prediction.
    Implements returning a sliding temporal window of frames for causal modeling.
    Includes a robust caching mechanism to load annotations instantly after the first run.
    """
    def __init__(self, dataset_dir, split='train', cache_dir="./data/cache", transform=None, window_size=8, multi_crop=False):
        """
        Args:
            dataset_dir (str): Root directory to the CholecT45 dataset (e.g. /raid/...)
            split (str): One of ['train', 'val', 'test']
            cache_dir (str): Path to save the compiled dataset index
            transform: torchvision transforms
            window_size (int): Number of consecutive frames to return (T). Default: 8
            multi_crop (bool): If True, applies 10-crop testing transforms (for eval usually)
        """
        self.dataset_dir = dataset_dir
        self.cache_dir = cache_dir
        self.transform = transform
        self.window_size = window_size
        self.multi_crop = multi_crop
        self.split = split
        
        # Path to our saved parsed cache
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_path = os.path.join(self.cache_dir, "cholect45_parsed_index.pkl")
        
        # Load all samples, then filter by split
        all_samples = self._load_or_build_index()
        self.samples = self._filter_by_split(all_samples)
        print(f">>> [Dataset] Final count for {split} split: {len(self.samples)} frames.")

    def _filter_by_split(self, all_samples):
        # We manually define splits based on sorted video IDs to ensure reproducibility
        # Total 45 videos. We use 35 for training, 5 for val, 5 for test.
        video_dirs = sorted(list(set([s['video_id'] for s in all_samples])))
        
        if self.split == 'train':
            split_vids = video_dirs[:35]
        elif self.split == 'val':
            split_vids = video_dirs[35:40]
        elif self.split == 'test':
            split_vids = video_dirs[40:]
        else:
            raise ValueError("Split must be 'train', 'val', or 'test'")
            
        filtered = [s for s in all_samples if s['video_id'] in split_vids]
        
        # Re-index video_start_idx correctly after filtering
        # The original video_start_idx references the all_samples list.
        # We need it relative to the new filtered list.
        if filtered:
            current_video_id = None
            current_start_idx = 0
            for i, s in enumerate(filtered):
                if s['video_id'] != current_video_id:
                    current_video_id = s['video_id']
                    current_start_idx = i
                s['video_start_idx'] = current_start_idx
                
        return filtered

    def _load_or_build_index(self):
        """
        Checks if the parsed index already exists. If yes, load it instantly.
        If no, scan all video txt files, parse the float logic, build a list of all frames,
        and save it out. This satisfies the strict optimization requirement!
        """
        if os.path.exists(self.cache_path):
            print(f">>> [Dataset] Loading cached index from {self.cache_path}...")
            with open(self.cache_path, "rb") as f:
                samples = pickle.load(f)
            print(f">>> [Dataset] Successfully loaded {len(samples)} total frames from cache.")
            return samples
        
        print(f">>> [Dataset] Cache not found at {self.cache_path}. Building from scratch... (this might take a minute)")
        
        # Structure: <dataset_dir>/data/<video_id>/<frame_id>.png
        video_dirs = sorted([d for d in os.listdir(os.path.join(self.dataset_dir, "data")) if d.startswith("VID")])
        
        samples = []
        global_idx = 0
        
        for vid in video_dirs:
            print(f"    -> Parsing {vid}...")
            img_dir = os.path.join(self.dataset_dir, "data", vid)
            # Use natural sorting if possible, but sorted() is fine for pad-zero IDs
            frame_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
            
            try:
                triplet_path = os.path.join(self.dataset_dir, "triplet", f"{vid}.txt")
                instrument_path = os.path.join(self.dataset_dir, "instrument", f"{vid}.txt")
                verb_path = os.path.join(self.dataset_dir, "verb", f"{vid}.txt")
                target_path = os.path.join(self.dataset_dir, "target", f"{vid}.txt")
                
                # We use numpy to quickly parse these comma-separated lines.
                # Use fast processing: skip header if exists. CholecT45 txts usually don't have headers but row IDs.
                triplet_data = np.loadtxt(triplet_path, delimiter=",", dtype=np.float32)
                instrument_data = np.loadtxt(instrument_path, delimiter=",", dtype=np.float32)
                verb_data = np.loadtxt(verb_path, delimiter=",", dtype=np.float32)
                target_data = np.loadtxt(target_path, delimiter=",", dtype=np.float32)
            except Exception as e:
                print(f"Warning: Missing or malformed label file for {vid}. Error: {e}")
                continue
                
            # Track start index for this specific video for causal window clamping
            video_start_idx_in_filtered_list = 0 # This will be set per item relative to filtered segment
            
            # Temporary list for this video to calculate local offsets
            video_samples = []
            for i, frame_file in enumerate(frame_files):
                if i >= len(triplet_data):
                    break 
                    
                sample_dict = {
                    'img_path': os.path.join(img_dir, frame_file),
                    'video_id': vid,
                    'frame_id': i,
                    'video_start_idx': -1, # Set later
                    'instrument': instrument_data[i, 1:],
                    'verb': verb_data[i, 1:],
                    'target': target_data[i, 1:],
                    'triplet': triplet_data[i, 1:]
                }
                video_samples.append(sample_dict)
            
            # Now add this video's samples to global list
            v_start = len(samples)
            for s in video_samples:
                s['video_start_idx'] = v_start
                samples.append(s)
                
        # Cache it for next time
        print(f">>> [Dataset] Saving parsed cache to {self.cache_path}...")
        with open(self.cache_path, "wb") as f:
            pickle.dump(samples, f)
            
        print(f">>> [Dataset] Dataset building complete. Total frames encoded: {len(samples)}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieves a temporal window of frames and the ground truth for the *last* frame.
        We simulate a causal (no-future) approach by taking [idx - window_size + 1, ..., idx].
        """
        current_sample = self.samples[idx]
        video_start_idx = current_sample['video_start_idx']
        
        frames = []
        
        # Optimization: Pre-calculate indices to avoid repeated logic in loop
        indices = [max(idx - i, video_start_idx) for i in range(self.window_size - 1, -1, -1)]
        
        for target_idx in indices:
            frame_path = self.samples[target_idx]['img_path']
            # Fast Loading
            img = Image.open(frame_path).convert('RGB')
            
            if self.transform is not None:
                img = self.transform(img)
                
            frames.append(img)
            
        # Stack into (T, C, H, W)
        frames_tensor = torch.stack(frames, dim=0)
        
        # Ground truths for the current frame
        instrument_gt = torch.tensor(current_sample['instrument'], dtype=torch.float32)
        verb_gt = torch.tensor(current_sample['verb'], dtype=torch.float32)
        target_gt = torch.tensor(current_sample['target'], dtype=torch.float32)
        triplet_gt = torch.tensor(current_sample['triplet'], dtype=torch.float32)
        
        return frames_tensor, (instrument_gt, verb_gt, target_gt, triplet_gt, current_sample['video_id'])

def build_transforms(is_train=True):
    """
    Builds the PyTorch Image Transformations for CholecT45.
    """
    if is_train:
        # Optimization: Add slight augmentation to improve generalization
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

def get_dataloader(dataset_dir, split='train', batch_size=4, num_workers=16, window_size=8, pin_memory=False):
    """
    Constructs the standard DataLoader. 
    """
    transform = build_transforms(is_train=(split == 'train'))
    dataset = CholecT45Dataset(dataset_dir, split=split, transform=transform, window_size=window_size)
    
    dl = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=(split == 'train'), 
        num_workers=num_workers,
        pin_memory=pin_memory 
    )
    
    return dl, dataset

if __name__ == "__main__":
    # Test block to verify our dataloader and caching works properly!
    print("--- Testing Dataset & Caching ---")
    TEST_DATASET_DIR = "/raid/manoranjan/rampreetham/CholecT45"
    
    dl, ds = get_dataloader(TEST_DATASET_DIR, batch_size=2, num_workers=4)
    
    for batch_idx, (frames, labels) in enumerate(dl):
        inst, verb, target, trip, vid = labels
        print(f"Batch {batch_idx}:")
        print(f" - Frames Shape: {frames.shape} (Expected: B, T, C, H, W)")
        print(f" - Instrument GT Shape: {inst.shape}")
        print(f" - Verb GT Shape: {verb.shape}")
        print(f" - Target GT Shape: {target.shape}")
        print(f" - Triplet GT Shape: {trip.shape}")
        break # Test one batch and exit cleanly

    print("--- Test Passed ---")

```

---

## File: `src/models/backbone.py`

```python
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

```

---

## File: `src/models/query_decoder.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BipartiteBindingDecoder(nn.Module):
    """
    Decoupled Multi-Task Query Decoder with Bipartite Binding.
    
    1. Instrument queries (Q_I) and Target queries (Q_T) decode from the visual features.
    2. Q_I and Q_T perform cross-attention (Bipartite Binding) to establish physical relations.
    3. Verb queries (Q_V) decode from the bound (Q_I, Q_T) representations to predict verbs 
       conditioned on the localized instruments and targets.
    """
    def __init__(self, d_model=1024, num_queries=64, num_layers=2, nhead=8):
        super(BipartiteBindingDecoder, self).__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        
        # Learnable Queries (num_queries per task)
        self.query_I = nn.Parameter(torch.randn(1, num_queries, d_model))
        self.query_T = nn.Parameter(torch.randn(1, num_queries, d_model))
        self.query_V = nn.Parameter(torch.randn(1, num_queries, d_model))
        
        # Transformer Decoders for initial localized grounding
        decoder_layer_I = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.visual_decoder_I = nn.TransformerDecoder(decoder_layer_I, num_layers=num_layers)
        
        decoder_layer_T = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.visual_decoder_T = nn.TransformerDecoder(decoder_layer_T, num_layers=num_layers)
        
        # Bipartite Binding Layer (Instrument <-> Target Cross Attention)
        # Using a transformer decoder layer where Q_I is query and Q_T is Mem (and vice-versa optionally)
        self.binding_I2T = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.binding_T2I = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm_I = nn.LayerNorm(d_model)
        self.norm_T = nn.LayerNorm(d_model)
        
        # Verb Decoder (looks at bound features instead of raw visual features)
        # We concatenate I and T as the memory memory bank for V
        decoder_layer_V = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.verb_decoder = nn.TransformerDecoder(decoder_layer_V, num_layers=num_layers)
        
        # Final classification heads
        # Assume cholecT45 output dimensions based on labels:
        # Instrument (6), Verb (10), Target (15)
        self.head_I = nn.Linear(d_model, 6)
        self.head_V = nn.Linear(d_model, 10)
        self.head_T = nn.Linear(d_model, 15)

    def forward(self, visual_features):
        """
        Args:
            visual_features: (B, SeqLen, d_model). Can be spatial or temporal sequence.
                             Usually we take the target temporal frame (Current frame) 
                             or the sequence. For this design, we treat the sequence as memory.
        Returns:
            logits_I: (B, 6)
            logits_V: (B, 10)
            logits_T: (B, 15)
            z_I, z_T, z_V: Tensors for contrastive loss memory banks
        """
        B = visual_features.shape[0]
        
        # Expand queries for batch
        q_I = self.query_I.expand(B, -1, -1) # (B, num_queries, d_model)
        q_T = self.query_T.expand(B, -1, -1)
        q_V = self.query_V.expand(B, -1, -1)
        
        # 1. Ground queries directly to visuals
        out_I = self.visual_decoder_I(q_I, visual_features) # (B, Q, d)
        out_T = self.visual_decoder_T(q_T, visual_features) # (B, Q, d)
        
        # 2. Bipartite Binding
        # I attends to T
        bound_I, _ = self.binding_I2T(out_I, out_T, out_T)
        out_I = self.norm_I(out_I + bound_I)
        
        # T attends to I
        bound_T, _ = self.binding_T2I(out_T, out_I, out_I)
        out_T = self.norm_T(out_T + bound_T)
        
        # 3. Verb Decoding from joint semantic space 
        # Create a joint memory of established Instrument & Target presence
        joint_memory = torch.cat([out_I, out_T], dim=1) # (B, 2*Q, d)
        out_V = self.verb_decoder(q_V, joint_memory)
        
        # We max-pool across query tokens to get video-level logits for the frame
        # (B, Q, d) -> (B, d)
        z_I = out_I.max(dim=1)[0]
        z_T = out_T.max(dim=1)[0]
        z_V = out_V.max(dim=1)[0]
        
        # Compute final class logits
        logits_I = self.head_I(z_I)
        logits_T = self.head_T(z_T)
        logits_V = self.head_V(z_V)
        
        # We return latent embeddings (z) as well, as they are needed for the 
        # Tail-Boosted Contrastive Memory Loss (SupCon).
        return logits_I, logits_V, logits_T, (z_I, z_V, z_T)

if __name__ == "__main__":
    print("--- Testing Query Decoder ---")
    model = BipartiteBindingDecoder().cuda()
    
    # Dummy T-Encoder Output: (Batch, TemporalSeq, d_model) -> (2, 8, 1024)
    dummy_mem = torch.randn(2, 8, 1024).cuda()
    
    i, v, t, z = model(dummy_mem)
    print(f"Logits I: {i.shape} (Expected: B, 6)")
    print(f"Logits V: {v.shape} (Expected: B, 10)")
    print(f"Logits T: {t.shape} (Expected: B, 15)")
    print(f"Latent I: {z[0].shape} (Expected: B, 1024)")
    print("--- Test Passed ---")

```

---

## File: `src/models/refiner.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DenoisingSemanticRefiner(nn.Module):
    """
    Denoising Semantic Refiner (DSR).
    
    Acts as a simulated discrete diffusion module to 'denoise' the initial 
    multi-label predictions into anatomically correct Triplets.
    
    Prevents hallucination by constraining outputs using clinical knowledge matrices.
    """
    def __init__(self, step_count=3, dim_hidden=256, num_I=6, num_V=10, num_T=15, num_triplets=100):
        super(DenoisingSemanticRefiner, self).__init__()
        self.step_count = step_count
        self.num_I = num_I
        self.num_V = num_V
        self.num_T = num_T
        self.num_triplets = num_triplets
        
        # Total semantic input dim
        dim_in = num_I + num_V + num_T
        
        # Iterative MLP network that updates the joint probability distribution
        self.refine_mlp = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.GELU(),
            nn.LayerNorm(dim_hidden),
            nn.Linear(dim_hidden, dim_hidden),
            nn.GELU(),
            nn.LayerNorm(dim_hidden),
            nn.Linear(dim_hidden, dim_in)
        )
        
        # Knowledge-Guided Mask: Conditional Probability Matrix P(V|I,T)
        # We initialize as uniform prior, then attempt to load precomputed stats.
        self.register_buffer("conditional_prior", torch.ones(num_I, num_T, num_V) / num_V)
        
        # Load from stats if available (Compute once, reuse many times)
        import json
        import os
        stats_path = "data/cache/stats.json"
        if os.path.exists(stats_path):
            try:
                with open(stats_path, "r") as f:
                    stats = json.load(f)
                prior = torch.tensor(stats['conditional_prior'], dtype=torch.float32)
                self.conditional_prior.copy_(prior)
                print(f">>> [DSR] Loaded precomputed clinical prior from {stats_path}")
            except Exception as e:
                print(f">>> [DSR] Warning: Could not load clinical prior: {e}")
        
        # Final Triplet Projection
        self.triplet_proj = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.GELU(),
            nn.Linear(dim_hidden, num_triplets)
        )

    def update_prior(self, prior_tensor):
        """
        Updates the internal knowledge matrix from empirical dataset stats.
        Expected tensor shape: (6, 15, 10)
        """
        self.conditional_prior.copy_(prior_tensor)

    def forward(self, logits_I, logits_V, logits_T):
        """
        Args:
            logits_I: (B, 6)
            logits_V: (B, 10)
            logits_T: (B, 15)
        Returns:
            ref_I, ref_V, ref_T: Refined multi-label logits
            triplet_logits: (B, 100) The final triplet prediction
        """
        with torch.amp.autocast('cuda', enabled=False):
            logits_I = logits_I.float()
            logits_V = logits_V.float()
            logits_T = logits_T.float()
            
            # Form initial "noisy" state
            B = logits_I.shape[0]
            x_t = torch.cat([logits_I, logits_V, logits_T], dim=1) # (B, 31)
            
            for step in range(self.step_count):
                # Propose delta
                delta = self.refine_mlp(x_t)
                
                # Simulated Denoising update
                x_t = x_t + delta
                
                # Apply logical constraint / Masking
                # We scale the verb probabilities based on the max probability of I, T pairs
                # Note: For extreme strictness during backward, we can use the prior to mask 
                # unreachable verbs, but here we do it softly using continuous relaxations.
                
                # Multi-label support: Use sigmoid instead of softmax because multiple tools/targets 
                # can be present in a frame.
                curr_I = x_t[:, :self.num_I].sigmoid()
                curr_V = x_t[:, self.num_I:self.num_I+self.num_V]
                curr_T = x_t[:, self.num_I+self.num_V:].sigmoid()
                
                # Simple soft-masking heuristic: 
                # Expected Verb | I, T = sum_i sum_t P(I=i)P(T=t) * Prior(V|i,t)
                # einsum formulation: b i, b t, i t v -> b v
                expected_V_mask = torch.einsum('bi,bt,itv->bv', curr_I, curr_T, self.conditional_prior.to(curr_I.device))
                
                # Guiding V logits with logical correction. 
                # Increase epsilon to 1e-4 for mixed-precision stability (prevents 1/x gradient explosion)
                v_correction = torch.log(expected_V_mask + 1e-4)
                
                # Clamp correction to prevent extreme logit shifts
                v_correction = torch.clamp(v_correction, min=-10.0, max=10.0)
                
                # Update the V region of x_t
                x_t[:, self.num_I:self.num_I+self.num_V] += 0.1 * v_correction
                
            # The refined state is now coherent.
            ref_I = x_t[:, :self.num_I]
            ref_V = x_t[:, self.num_I:self.num_I+self.num_V]
            ref_T = x_t[:, self.num_I+self.num_V:]
            
            # Project refined joint probabilities to direct Triplet labels for ASL loss
            triplet_logits = self.triplet_proj(x_t)
            
        return ref_I, ref_V, ref_T, triplet_logits

if __name__ == "__main__":
    print("--- Testing Denoising Semantic Refiner ---")
    model = DenoisingSemanticRefiner().cuda()
    
    # Simulating outputs from the Query Decoder
    dummy_I = torch.randn(2, 6).cuda()
    dummy_V = torch.randn(2, 10).cuda()
    dummy_T = torch.randn(2, 15).cuda()
    
    r_I, r_V, r_T, r_Trip = model(dummy_I, dummy_V, dummy_T)
    print(f"Refined I: {r_I.shape}")
    print(f"Refined V: {r_V.shape}")
    print(f"Refined T: {r_T.shape}")
    print(f"Triplet Logits: {r_Trip.shape} (Expected: B, 100)")
    print("--- Test Passed ---")

```

---

## File: `src/losses/asl.py`

```python
import torch
import torch.nn as nn

class AsymmetricLossOptimized(nn.Module):
    """
    Asymmetric Loss (ASL) for Multi-Label Classification.
    Provides better handling of positive/negative imbalance than standard BCE
    by dynamically down-weighting easy negatives and applying hard thresholds.
    
    Formula based on: 'Asymmetric Loss For Multi-Label Classification' (ICCV 2021)
    """
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.0, eps=1e-8, reduce='mean'):
        super(AsymmetricLossOptimized, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduce = reduce

    def forward(self, x, y):
        """
        Args:
            x: input logits (B, num_classes)
            y: targets (B, num_classes), boolean multi-labels (0 or 1)
        """
        # Cast to float32 to prevent FP16 underflow leading to NaN from log(0) and 0 * -inf
        x = x.float()
        y = y.float()

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))

        # Asymmetric Focusing
        loss = los_pos * ((1 - xs_pos).clamp(min=self.eps)) ** self.gamma_pos + \
               los_neg * (xs_neg.clamp(min=self.eps)) ** self.gamma_neg

        # Negative sign to make it a loss to minimize
        loss = -loss

        if self.reduce == 'mean':
            return loss.mean()
        elif self.reduce == 'sum':
            return loss.sum()
        else:
            return loss

if __name__ == "__main__":
    print("--- Testing ASL Loss ---")
    loss_fn = AsymmetricLossOptimized()
    
    dummy_logits = torch.randn(2, 100)
    dummy_targets = torch.randint(0, 2, (2, 100)).float()
    
    loss = loss_fn(dummy_logits, dummy_targets)
    print(f"ASL Loss output: {loss.item()}")
    print("--- Test Passed ---")

```

---

## File: `src/losses/mcl.py`

```python
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

