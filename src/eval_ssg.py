import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import average_precision_score

def evaluate_sg_model(model, dataloader, device, logger, is_sample_run=False):
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
            
            if is_sample_run and batch_idx >= 5:
                logger.info("Sample run: Breaking eval loop early for quick test.")
                break
            
    # Calculate mAP for edges
    all_edge_preds = np.vstack(all_edge_preds)
    all_edge_gts = np.vstack(all_edge_gts)
    
    # Secure against stray NaNs in predictions breaking sklearn metrics
    all_edge_preds = np.nan_to_num(all_edge_preds, nan=0.0)
    all_edge_gts = np.nan_to_num(all_edge_gts, nan=0.0)
    
    edge_map = average_precision_score(all_edge_gts, all_edge_preds, average='macro')
    energy_acc = energy_correct / energy_total if energy_total > 0 else 0.0
    
    logger.info(f"--- SSG Evaluation Results ---")
    logger.info(f"Edge Prediction mAP: {edge_map:.4f}")
    logger.info(f"Energy State Acc:    {energy_acc:.4f}")
    
    return edge_map

if __name__ == "__main__":
    print("--- Eval SSG module loaded ---")
