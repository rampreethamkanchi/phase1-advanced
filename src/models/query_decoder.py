import torch
import torch.nn as nn
import torch.nn.functional as F

class BipartiteBindingDecoder(nn.Module):
    """
    Decoupled Multi-Task Query Decoder with Bipartite Binding.
    
    1. Instrument queries (Q_I) and Target queries (Q_T) decode from the visual features.
    2. Q_I and Q_T perform cross-attention (Bipartite Binding) to establish physical relations.
    3. Verb queries (Q_V) decode from the bound (Q_I, Q_T) representations to predict verbs 
       conditioned on the localized instruments and targets.
    """
    def __init__(self, d_model=1024, num_queries=64, num_layers=2, nhead=8):
        super(BipartiteBindingDecoder, self).__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        
        # Learnable Queries (num_queries per task)
        self.query_I = nn.Parameter(torch.randn(1, num_queries, d_model))
        self.query_T = nn.Parameter(torch.randn(1, num_queries, d_model))
        self.query_V = nn.Parameter(torch.randn(1, num_queries, d_model))
        
        # Transformer Decoders for initial localized grounding
        decoder_layer_I = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.visual_decoder_I = nn.TransformerDecoder(decoder_layer_I, num_layers=num_layers)
        
        decoder_layer_T = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.visual_decoder_T = nn.TransformerDecoder(decoder_layer_T, num_layers=num_layers)
        
        # Bipartite Binding Layer (Instrument <-> Target Cross Attention)
        # Using a transformer decoder layer where Q_I is query and Q_T is Mem (and vice-versa optionally)
        self.binding_I2T = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.binding_T2I = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm_I = nn.LayerNorm(d_model)
        self.norm_T = nn.LayerNorm(d_model)
        
        # Verb Decoder (looks at bound features instead of raw visual features)
        # We concatenate I and T as the memory memory bank for V
        decoder_layer_V = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.verb_decoder = nn.TransformerDecoder(decoder_layer_V, num_layers=num_layers)
        
        # Final classification heads
        # Assume cholecT45 output dimensions based on labels:
        # Instrument (6), Verb (10), Target (15)
        self.head_I = nn.Linear(d_model, 6)
        self.head_V = nn.Linear(d_model, 10)
        self.head_T = nn.Linear(d_model, 15)

    def forward(self, visual_features):
        """
        Args:
            visual_features: (B, SeqLen, d_model). Can be spatial or temporal sequence.
                             Usually we take the target temporal frame (Current frame) 
                             or the sequence. For this design, we treat the sequence as memory.
        Returns:
            logits_I: (B, 6)
            logits_V: (B, 10)
            logits_T: (B, 15)
            z_I, z_T, z_V: Tensors for contrastive loss memory banks
        """
        B = visual_features.shape[0]
        
        # Expand queries for batch
        q_I = self.query_I.expand(B, -1, -1) # (B, num_queries, d_model)
        q_T = self.query_T.expand(B, -1, -1)
        q_V = self.query_V.expand(B, -1, -1)
        
        # 1. Ground queries directly to visuals
        out_I = self.visual_decoder_I(q_I, visual_features) # (B, Q, d)
        out_T = self.visual_decoder_T(q_T, visual_features) # (B, Q, d)
        
        # 2. Bipartite Binding
        # I attends to T
        bound_I, _ = self.binding_I2T(out_I, out_T, out_T)
        out_I = self.norm_I(out_I + bound_I)
        
        # T attends to I
        bound_T, _ = self.binding_T2I(out_T, out_I, out_I)
        out_T = self.norm_T(out_T + bound_T)
        
        # 3. Verb Decoding from joint semantic space 
        # Create a joint memory of established Instrument & Target presence
        joint_memory = torch.cat([out_I, out_T], dim=1) # (B, 2*Q, d)
        out_V = self.verb_decoder(q_V, joint_memory)
        
        # We max-pool across query tokens to get video-level logits for the frame
        # (B, Q, d) -> (B, d)
        z_I = out_I.max(dim=1)[0]
        z_T = out_T.max(dim=1)[0]
        z_V = out_V.max(dim=1)[0]
        
        # Compute final class logits
        logits_I = self.head_I(z_I)
        logits_T = self.head_T(z_T)
        logits_V = self.head_V(z_V)
        
        # We return latent embeddings (z) as well, as they are needed for the 
        # Tail-Boosted Contrastive Memory Loss (SupCon).
        return logits_I, logits_V, logits_T, (z_I, z_V, z_T)

if __name__ == "__main__":
    print("--- Testing Query Decoder ---")
    model = BipartiteBindingDecoder().cuda()
    
    # Dummy T-Encoder Output: (Batch, TemporalSeq, d_model) -> (2, 8, 1024)
    dummy_mem = torch.randn(2, 8, 1024).cuda()
    
    i, v, t, z = model(dummy_mem)
    print(f"Logits I: {i.shape} (Expected: B, 6)")
    print(f"Logits V: {v.shape} (Expected: B, 10)")
    print(f"Logits T: {t.shape} (Expected: B, 15)")
    print(f"Latent I: {z[0].shape} (Expected: B, 1024)")
    print("--- Test Passed ---")
