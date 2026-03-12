# Methodology Context: Phase 3 Scene Graphs

This document contains the complete source code for all modules contributing to this phase. Use this as context for writing the methodology section of the research paper.

## File: `src/dataset_ssg.py`

```python
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pickle
import numpy as np

class SSGVQADataset(Dataset):
    """
    Dataset for SSG-VQA (Surgical Scene Graphs).
    Extracts Nodes (Instruments/Anatomy), Edges (Relationships), and Phase 3 Attributes.
    """
    def __init__(self, dataset_dir, split='train', cache_dir="./data/cache", transform=None, window_size=8):
        self.dataset_dir = dataset_dir
        self.cache_dir = cache_dir
        self.transform = transform
        self.window_size = window_size
        self.split = split
        
        # Vocabularies derived from SSG-VQA standards
        self.node_classes = [
            'grasper', 'bipolar', 'hook', 'scissors', 'clipper', 'irrigator', 
            'liver', 'gallbladder', 'cystic_plate', 'cystic_duct', 'cystic_artery', 
            'cystic_pedicle', 'blood_vessel', 'fluid', 'abdominal_wall_cavity', 
            'omentum', 'gut', 'specimen'
        ]
        self.edge_classes = [
            'grasp', 'retract', 'dissect', 'coagulate', 'clip', 'cut', 'aspirate', 'wash', 'null',
            'above', 'below', 'left', 'right', 'horizontal', 'vertical', 'within', 'out_of', 'surround'
        ]
        
        self.node2id = {v: i for i, v in enumerate(self.node_classes)}
        self.edge2id = {v: i for i, v in enumerate(self.edge_classes)}
        
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_path = os.path.join(self.cache_dir, f"ssg_parsed_{split}.pkl")
        
        self.samples = self._load_or_build_index()

    def _get_split_videos(self):
        # Using standard 50-video split logic matching CholecT50 roughly
        # If SSG-VQA has specific split files we would load them here.
        # According to the sample JSON ("split": "new"), we just parse all for demo, 
        # or separate by ID.
        all_videos = sorted(list(set([d.split('_')[0] for d in os.listdir(os.path.join(self.dataset_dir, "scene_graph/scene_graph")) if d.endswith('.json')])))
        if self.split == 'train':
            return all_videos[:35]
        elif self.split == 'val':
            return all_videos[35:40]
        elif self.split == 'test':
            return all_videos[40:]
        else:
            return all_videos

    def _load_or_build_index(self):
        if os.path.exists(self.cache_path):
            print(f">>> [Dataset] Loading cached SSG-VQA index from {self.cache_path}...")
            with open(self.cache_path, "rb") as f:
                samples = pickle.load(f)
            return samples
        
        print(f">>> [Dataset] Building SSG-VQA index for {self.split} split...")
        split_vids = self._get_split_videos()
        samples = []
        
        sg_dir = os.path.join(self.dataset_dir, "scene_graph/scene_graph")
        img_dir_base = os.path.join(self.dataset_dir, "visual_feats", "images")
        
        all_json_files = sorted([f for f in os.listdir(sg_dir) if f.endswith('.json')])
        
        current_vid = None
        v_start = 0
        
        from tqdm import tqdm
        for json_file in tqdm(all_json_files, desc="Parsing Scene Graphs"):
            vid_id = json_file.split('_')[0]
            if vid_id not in split_vids:
                continue
                
            if current_vid != vid_id:
                current_vid = vid_id
                v_start = len(samples)
                
            with open(os.path.join(sg_dir, json_file), "r") as f:
                d = json.load(f)
                
            for scene in d.get("scenes", []):
                frame_name = scene.get("image_filename", "")
                # e.g., VID12_0123.png
                vid_folder = frame_name.split('_')[0]
                img_path = os.path.join(img_dir_base, vid_folder, f"{frame_name}.png")
                
                # Parse Nodes
                nodes = []
                bboxes = []
                for obj in scene.get("objects", []):
                    cls_name = obj.get("component", "null")
                    n_id = self.node2id.get(cls_name, -1)
                    nodes.append(n_id)
                    bboxes.append(obj.get("bbox", [0, 0, 0, 0]))
                
                # Parse Edges (Relationships)
                # Adjacency Matrix: [Num_Nodes, Num_Nodes, Num_Edge_Classes]
                num_nodes = len(nodes)
                edge_matrix = np.zeros((num_nodes, num_nodes, len(self.edge_classes)), dtype=np.float32)
                
                rels = scene.get("relationships", {})
                for rel_name, rel_list in rels.items():
                    e_id = self.edge2id.get(rel_name, -1)
                    if e_id == -1: continue
                        
                    for subj_idx, targets in enumerate(rel_list):
                        for obj_idx in targets:
                            if subj_idx < num_nodes and obj_idx < num_nodes:
                                edge_matrix[subj_idx, obj_idx, e_id] = 1.0
                
                # Plan A: Attribute Enrichment Defaults
                # Since SSG-VQA native doesn't explicitly label state/proximity in numbers, 
                # we generate 'silver' dummy targets for our auxiliary heads based on relations
                
                # Active Energy: True if predicate is dissect, coagulate, cut
                active_energy = 0.0
                active_rels = ['dissect', 'coagulate', 'cut']
                for a_r in active_rels:
                    if a_r in rels and any(len(t) > 0 for t in rels[a_r]):
                        active_energy = 1.0
                        break
                        
                samples.append({
                    'img_path': img_path,
                    'video_id': vid_id,
                    'frame_id': int(frame_name.split('_')[1]) if '_' in frame_name else 0,
                    'video_start_idx': v_start,
                    'nodes': nodes,
                    'bboxes': bboxes,
                    'edges': edge_matrix,
                    'active_energy': active_energy
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
            # Fallback if SSG images aren't extracted yet (since they depend on Cholec80 raw videos)
            if not os.path.exists(frame_path):
                 # Create dummy frame for testing logic
                 img = Image.new('RGB', (448, 448))
            else:
                 img = Image.open(frame_path).convert('RGB')
                 
            if self.transform is not None:
                img = self.transform(img)
            frames.append(img)
            
        frames_tensor = torch.stack(frames, dim=0)
        
        # We cap nodes at 15 for batching padding
        MAX_NODES = 15
        nodes = current_sample['nodes'][:MAX_NODES]
        bboxes = current_sample['bboxes'][:MAX_NODES]
        # Pad nodes
        padded_nodes = np.ones(MAX_NODES, dtype=np.int64) * -1
        padded_bboxes = np.zeros((MAX_NODES, 4), dtype=np.float32)
        padded_nodes[:len(nodes)] = nodes
        padded_bboxes[:len(nodes)] = bboxes
        
        # Pad Edges
        # Edges -> (MAX_NODES, MAX_NODES, NUM_EDGE_CLASSES)
        padded_edges = np.zeros((MAX_NODES, MAX_NODES, len(self.edge_classes)), dtype=np.float32)
        e_mat = current_sample['edges']
        n_m = min(e_mat.shape[0], MAX_NODES)
        padded_edges[:n_m, :n_m, :] = e_mat[:n_m, :n_m, :]
        
        # Energy Attr
        active_energy = torch.tensor(current_sample['active_energy'], dtype=torch.float32)

        out_dict = {
            'video_id': current_sample['video_id'],
            'nodes': torch.tensor(padded_nodes),
            'bboxes': torch.tensor(padded_bboxes),
            'edges': torch.tensor(padded_edges),
            'active_energy': active_energy,
            'num_valid_nodes': torch.tensor(len(nodes))
        }
        
        return frames_tensor, out_dict

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

def get_dataloader_ssg(dataset_dir, split='train', batch_size=4, num_workers=16, window_size=8, pin_memory=False):
    transform = build_transforms(is_train=(split == 'train'))
    dataset = SSGVQADataset(dataset_dir, split=split, transform=transform, window_size=window_size)
    
    dl = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=(split == 'train'), 
        num_workers=num_workers,
        pin_memory=pin_memory 
    )
    
    return dl, dataset

if __name__ == "__main__":
    
    print("--- Testing Dataset & Caching (SSG-VQA) ---")
    TEST_DATASET_DIR = "/raid/manoranjan/rampreetham/SSG-VQA"
    
    dl, ds = get_dataloader_ssg(TEST_DATASET_DIR, split='val', batch_size=2, num_workers=4)
    
    for batch_idx, (frames, labels) in enumerate(dl):
        print(f"Batch {batch_idx}:")
        print(f" - Frames Shape: {frames.shape} (Expected: B, T, C, H, W)")
        print(f" - Nodes Shape: {labels['nodes'].shape} (Expected: B, MAX_NODES)")
        print(f" - Edges Shape: {labels['edges'].shape} (Expected: B, MAX_NODES, MAX_NODES, NUM_EDGE_CLASSES)")
        print(f" - Active Energy Shape: {labels['active_energy'].shape}")
        break 

```

---

## File: `src/train_ssg.py`

```python
import os
import time
import argparse
import datetime
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Import Custom Modules
from dataset_ssg import get_dataloader_ssg
from models.tdt import TriDiffTransformer
from losses.ssg_loss import SceneGraphLoss
from eval_ssg import evaluate_sg_model

def setup_logger(log_dir="./logs"):
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"ssg_run_{ts}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()
    return logger, log_file

def train_one_epoch(epoch, model, dataloader, optimizer, criterion, device, logger, scaler, args):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} (SSG Phase 3)")
    
    optimizer.zero_grad()
    
    for batch_idx, (frames, labels) in enumerate(progress_bar):
        frames = frames.to(device, non_blocking=True)
        nodes = labels['nodes'].to(device, non_blocking=True)
        bboxes = labels['bboxes'].to(device, non_blocking=True)
        edges_gt = labels['edges'].to(device, non_blocking=True)
        energy_gt = labels['active_energy'].to(device, non_blocking=True)
        num_valid_nodes = labels['num_valid_nodes'].to(device, non_blocking=True)
        
        # Mixed Precision
        with torch.amp.autocast('cuda', enabled=args.use_amp):
            # TDT returns: logits_I, logits_V, logits_T, triplet_logits, latent_z, spatial_feats, phase_logits, edge_logits, energy_logits
            _, _, _, _, _, _, _, edge_logits, energy_logits = model(frames, nodes, bboxes)
            
            loss_edge, loss_energy = criterion(edge_logits, edges_gt, energy_logits, energy_gt, num_valid_nodes)
            
            loss = (args.lam_edge * loss_edge) + (args.lam_energy * loss_energy)
            loss = loss / args.grad_accum_steps
             
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
             logger.info(f"[Epoch {epoch} | Batch {batch_idx}/{len(dataloader)}] Loss: {current_batch_loss:.4f} (Avg: {avg_loss_so_far:.4f}) -> [Training SG Relations]")
                         
        if args.sample_run and batch_idx >= 5:
             logger.info("Sample run: Breaking batch loop early.")
             break
                         
    avg_loss = total_loss / (batch_idx + 1)
    logger.info(f"*** Epoch {epoch} completed. Average Loss: {avg_loss:.4f} ***")
    return avg_loss

def main(args):
    logger, log_file = setup_logger()
    logger.info(f"Starting SSG-VQA Training Process. Plan: {args.plan}")
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    
    logger.info(f"Initializing DataLoaders for SSG-VQA from: {args.dataset_dir}")
    train_dl, _ = get_dataloader_ssg(args.dataset_dir, split='train', batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_memory)
    val_dl, _ = get_dataloader_ssg(args.dataset_dir, split='val', batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_memory)
    
    logger.info(f"Initializing TDT (Plan {args.plan}) for SSG computation...")
    model = TriDiffTransformer(use_refiner=True, plan=args.plan)
    
    # Optional: Load Phase 1/2 Checkpoint
    if args.pretrained_backbone:
        if os.path.exists(args.pretrained_backbone):
             logger.info(f"Loading Phase 1 weights from {args.pretrained_backbone}...")
             state_dict = torch.load(args.pretrained_backbone, map_location='cpu')
             model.load_state_dict(state_dict, strict=False)
        else:
             logger.warning(f"Could not find {args.pretrained_backbone}. Starting from scratch Swin.")
             
    model = model.to(device)
    
    # In Phase 3, we freeze the backbone and temporal layers and ONLY train the relational transformers
    if args.freeze_backbone:
        logger.info("Freezing Phase 1 spatial backbones to focus on Relational Training.")
        for param in model.backbone.parameters():
            param.requires_grad = False
        for param in model.t_encoder.parameters():
            param.requires_grad = False
        for param in model.decoder.parameters():
            param.requires_grad = False

    sg_params = list(model.relational_transformer.parameters()) if args.plan == 'A' else []
    
    optimizer = optim.AdamW(sg_params, lr=args.lr_sg, weight_decay=args.weight_decay)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    scaler = torch.amp.GradScaler('cuda', enabled=args.use_amp)
    criterion = SceneGraphLoss().to(device)
    
    best_mAP = -1.0
    for epoch in range(1, args.epochs + 1):
        logger.info(f"--- Launching SG Epoch {epoch}/{args.epochs} ---")
        
        train_loss = train_one_epoch(
            epoch, model, train_dl, optimizer, criterion, device, logger, scaler, args
        )
        
        scheduler.step()
        
        logger.info(f"--- Running SSG Evaluation for Epoch {epoch} ---")
        eval_metric = evaluate_sg_model(model, val_dl, device, logger)
        
        if args.sample_run and epoch == 2:
            break
            
        is_best = eval_metric > best_mAP
        if is_best:
            best_mAP = eval_metric
            best_path = f"logs/best_ssg_{args.plan}_model.pt"
            torch.save(model.state_dict(), best_path)
            logger.info(f"New Best SSG Edge mAP: {best_mAP:.4f} saved to {best_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Phase 3 SSG Training Script")
    parser.add_argument("--dataset_dir", type=str, default="/raid/manoranjan/rampreetham/SSG-VQA")
    parser.add_argument("--plan", type=str, default="A", choices=['A', 'B'])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr_sg", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    
    parser.add_argument("--lam_edge", type=float, default=1.0)
    parser.add_argument("--lam_energy", type=float, default=0.5)
    parser.add_argument("--pretrained_backbone", type=str, default="logs/best_model.pt", help="Path to best phase 1 model")
    parser.add_argument("--freeze_backbone", action="store_true", default=True)
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--pin_memory", action="store_true", default=False)
    
    parser.add_argument("--sample_run", action="store_true", help="Quick test")
    
    args = parser.parse_args()
    if args.plan == 'A':
         main(args)
    else:
         print("Plan B Fusion orchestrator not fully implemented for SSG training yet (relies on offline SOTA models). Use Plan A.")

```

---

## File: `src/losses/ssg_loss.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SceneGraphLoss(nn.Module):
    """
    Loss module for Phase 3 (Enhanced Surgical Scene Graph).
    Handles edge prediction (multi-label) and energy state prediction (binary).
    """
    def __init__(self, num_edge_classes=18):
        super().__init__()
        # We use BCEWithLogitsLoss because multiple relationships can exist between two nodes
        self.edge_criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.energy_criterion = nn.BCEWithLogitsLoss()
        
    def forward(self, edge_logits, edge_gt, energy_logits, energy_gt, num_valid_nodes=None):
        """
        edge_logits: (B, MAX_NODES, MAX_NODES, Num_Edge_Classes)
        edge_gt: (B, MAX_NODES, MAX_NODES, Num_Edge_Classes) - float32 multi-hot
        energy_logits: (B) Let's assume binary 'active' or not
        energy_gt: (B)
        num_valid_nodes: (B) -> True number of nodes per graph to ignore padded regions
        """
        B, N, _, C = edge_logits.shape
        
        # 1. Edge Loss
        raw_edge_loss = self.edge_criterion(edge_logits, edge_gt) # (B, N, N, C)
        
        # Mask out padded regions
        if num_valid_nodes is not None:
            mask = torch.zeros((B, N, N), device=edge_logits.device, dtype=torch.bool)
            for b in range(B):
                v_n = num_valid_nodes[b].item()
                mask[b, :v_n, :v_n] = True
                
            # Expand mask to classes
            mask = mask.unsqueeze(-1).expand_as(raw_edge_loss)
            
            # Compute mean only over valid pairs
            valid_loss = raw_edge_loss.masked_select(mask)
            edge_loss = valid_loss.mean() if valid_loss.numel() > 0 else torch.tensor(0.0).to(edge_logits.device)
        else:
            edge_loss = raw_edge_loss.mean()
            
        # 2. Energy Loss (Auxiliary Task from Phase 3 plan)
        energy_loss = self.energy_criterion(energy_logits, energy_gt)
        
        return edge_loss, energy_loss

```

---

## File: `src/eval_ssg.py`

```python
import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import average_precision_score

def evaluate_sg_model(model, dataloader, device, logger):
    """
    Evaluates Phase 3 Scene Graph model.
    Metrics:
    - Node Classification Accuracy (if applicable, though we provide nodes here)
    - Edge Prediction mAP (Relationship Matching)
    - Active Energy State Accuracy
    """
    model.eval()
    logger.info(f"Starting SSG Evaluation Pass...")
    
    all_edge_preds = []
    all_edge_gts = []
    
    energy_correct = 0
    energy_total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating SSG")
        for batch_idx, (frames, labels) in enumerate(progress_bar):
            frames = frames.to(device)
            nodes = labels['nodes'].to(device)
            bboxes = labels['bboxes'].to(device)
            edges_gt = labels['edges'].to(device)
            energy_gt = labels['active_energy'].to(device)
            num_valid_nodes = labels['num_valid_nodes'].to(device)
            
            # Forward pass
            _, _, _, _, _, _, _, edge_logits, energy_logits = model(frames, nodes, bboxes)
            
            # Edges
            edge_preds = torch.sigmoid(edge_logits) # (B, N, N, C)
            
            # Extract only valid pairs for mAP calculation
            B, N, _, C = edge_preds.shape
            for b in range(B):
                v_n = num_valid_nodes[b].item()
                if v_n > 0:
                    valid_preds = edge_preds[b, :v_n, :v_n, :].reshape(-1, C).cpu().numpy()
                    valid_gts = edges_gt[b, :v_n, :v_n, :].reshape(-1, C).cpu().numpy()
                    all_edge_preds.append(valid_preds)
                    all_edge_gts.append(valid_gts)
            
            # Energy
            energy_preds = torch.sigmoid(energy_logits) > 0.5
            energy_correct += (energy_preds.cpu() == energy_gt.cpu()).sum().item()
            energy_total += B
            
    # Calculate mAP for edges
    all_edge_preds = np.vstack(all_edge_preds)
    all_edge_gts = np.vstack(all_edge_gts)
    
    edge_map = average_precision_score(all_edge_gts, all_edge_preds, average='macro', zero_division=0)
    energy_acc = energy_correct / energy_total if energy_total > 0 else 0.0
    
    logger.info(f"--- SSG Evaluation Results ---")
    logger.info(f"Edge Prediction mAP: {edge_map:.4f}")
    logger.info(f"Energy State Acc:    {energy_acc:.4f}")
    
    return edge_map

if __name__ == "__main__":
    print("--- Eval SSG module loaded ---")

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

