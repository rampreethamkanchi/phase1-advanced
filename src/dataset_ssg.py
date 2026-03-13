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
        if len(nodes) > 0:
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
