import torch
import torch.nn as nn

class AsymmetricLossOptimized(nn.Module):
    """
    Asymmetric Loss (ASL) for Multi-Label Classification.
    Provides better handling of positive/negative imbalance than standard BCE
    by dynamically down-weighting easy negatives and applying hard thresholds.
    
    Formula based on: 'Asymmetric Loss For Multi-Label Classification' (ICCV 2021)
    """
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.0, eps=1e-8, reduce='mean'):
        super(AsymmetricLossOptimized, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduce = reduce

    def forward(self, x, y):
        """
        Args:
            x: input logits (B, num_classes)
            y: targets (B, num_classes), boolean multi-labels (0 or 1)
        """
        # Cast to float32 to prevent FP16 underflow leading to NaN from log(0) and 0 * -inf
        x = x.float()
        y = y.float()

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))

        # Asymmetric Focusing
        loss = los_pos * ((1 - xs_pos).clamp(min=self.eps)) ** self.gamma_pos + \
               los_neg * (xs_neg.clamp(min=self.eps)) ** self.gamma_neg

        # Negative sign to make it a loss to minimize
        loss = -loss

        if self.reduce == 'mean':
            return loss.mean()
        elif self.reduce == 'sum':
            return loss.sum()
        else:
            return loss

if __name__ == "__main__":
    print("--- Testing ASL Loss ---")
    loss_fn = AsymmetricLossOptimized()
    
    dummy_logits = torch.randn(2, 100)
    dummy_targets = torch.randint(0, 2, (2, 100)).float()
    
    loss = loss_fn(dummy_logits, dummy_targets)
    print(f"ASL Loss output: {loss.item()}")
    print("--- Test Passed ---")
