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
