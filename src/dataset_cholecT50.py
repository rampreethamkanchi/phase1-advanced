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
