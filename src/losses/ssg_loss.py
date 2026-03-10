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
