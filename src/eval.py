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
            logits_I, logits_V, logits_T, triplet_logits, _, _, phase_logits = model(frames)
            
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
