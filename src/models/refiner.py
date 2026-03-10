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
