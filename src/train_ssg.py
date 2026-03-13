import os
import sys
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
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.captureWarnings(True)
    logger = logging.getLogger()
    
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception
    
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
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
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
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
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
        eval_metric = evaluate_sg_model(model, val_dl, device, logger, is_sample_run=args.sample_run)
        
        if args.sample_run and epoch == 1:
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
    parser.add_argument("--pin_memory", action="store_true", default=True)
    
    parser.add_argument("--sample_run", action="store_true", help="Quick test")
    
    args = parser.parse_args()
    if args.plan == 'A':
         main(args)
    else:
         print("Plan B Fusion orchestrator not fully implemented for SSG training yet (relies on offline SOTA models). Use Plan A.")
