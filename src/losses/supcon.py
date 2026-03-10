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
