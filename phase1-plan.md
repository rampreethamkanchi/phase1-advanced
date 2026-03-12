This plan outlines the implementation of **Tri-Diff-Transformer (TDT)**, a novel architecture for surgical action triplet detection designed to exceed a 40% mAP. It integrates query-based localization, temporal query propagation, and a diffusion-inspired semantic refiner to specifically address the long-tail distribution of the CholecT45/50 dataset.

### 1. Architectural Strategy: Tri-Diff-Transformer (TDT)

The model moves away from global classification toward an **Object-Centric Query** approach, treating triplets as entities localized in space and time.

* **Backbone**: Swin-B (Base) Transformer pre-trained on Video-ImageNet.
* **Temporal Context**: A **Banded Causal Temporal Encoder**. Instead of standard LSTMs, it uses a transformer encoder with causal masks restricted to the last 10 frames to capture motion dynamics (e.g., the "cutting" motion) without looking into the future.
* **Decoupled Multi-Task Queries**:
* Initialize three sets of learnable queries: $Q_I$ (Instrument), $Q_V$ (Verb), and $Q_T$ (Target).
* **Bipartite Binding Layer**: A cross-attention module where $Q_I$ and $Q_T$ queries attend to each other to establish a spatial relationship before the $Q_V$ queries are applied to the joint feature.



### 2. Innovation: Denoising Semantic Refiner (DSR)

To overcome the 40% mAP ceiling and handle "spurious attributes" (where a model predicts a common verb because of a tool's appearance), we implement a **Discrete Diffusion-inspired Refiner**.

* **Logic**: Treat the initial triplet logits as "noisy" samples.
* **Mechanism**: A 3-step iterative MLP refines the logit distribution. In each step, it applies a **Knowledge-Guided Mask**. This mask is a pre-computed conditional probability matrix $P(\text{Verb} | \text{Instrument, Target})$ derived from the training set's clinical ground truth.
* **Benefit**: This prevents "hallucinated" triplets (e.g., a clipper performing a 'grasp' verb) which often degrade mAP in tail classes.

### 3. Solving Class Imbalance: Tail-Boosted Contrastive Memory (TBCM)

Drawing from Tail-Enhanced Representation Learning (TERL), we implement a memory-based contrastive branch.

* **Memory Bank**: A FIFO queue per tail class (bottom 15% of triplets). It stores the latent query embeddings $z$ of successfully detected rare triplets.
* **Supervised Contrastive Loss (SupCon)**:
* For every tail sample in a batch, pull $N$ positive samples from the memory bank.
* Minimize the distance between the current tail embedding and bank embeddings, while maximizing distance from "head" (common) classes.
* This forces the model to learn highly discriminative features for rare activities like "clipper-clip-cystic_artery".



### 4. Implementation Details for the Coding Agent

#### File Structure

```text
/src
  /models
    backbone.py (Swin-B)
    t_encoder.py (Banded Causal Transformer)
    query_decoder.py (MQ-DH with Bipartite Binding)
    refiner.py (Denoising Semantic Refiner)
  /losses
    asl.py (Asymmetric Multi-label Loss)
    supcon.py (Tail-specific Contrastive)
    mcl.py (Mutual-Channel Loss)
  train.py
  eval.py

```

#### Loss Function Formulation

The total loss $\mathcal{L}$ is a weighted sum:


$$\mathcal{L}_{total} = \mathcal{L}_{ASL} + 1.0 \cdot \mathcal{L}_{MCL} + 0.5 \cdot \mathcal{L}_{SupCon} + 0.2 \cdot \mathcal{L}_{DSR\_Reg}$$

1. **Asymmetric Loss (ASL)**: Replaces standard BCE to penalize easy negatives (null triplets) less, preventing them from overwhelming the gradient.
2. **Mutual-Channel Loss (MCL)**: Forces different feature channels to focus on different spatial regions (e.g., tool tip vs. tissue surface).
3. **DSR Regularization**: A KL-divergence loss to ensure the refined logits don't drift too far from the visual evidence.

#### Hyperparameters

* **Optimizer**: AdamW (Learning Rate: $1e-5$ for backbone, $1e-4$ for heads).
* **Input Size**: $384 \times 384$.
* **Temporal Window**: 8 frames.
* **Queries**: 64 per task (192 total).
* **Diffusion Steps**: 3.

### 5. Training and Validation Pipeline

1. **Phase 1 (Warmup)**: Train the backbone and query heads using only $\mathcal{L}_{ASL}$ for 10 epochs to stabilize localization.
2. **Phase 2 (Tail Boost)**: Enable the Memory Bank and $\mathcal{L}_{SupCon}$. Start oversampling batches to include at least 25% tail class samples.
3. **Phase 3 (Refinement)**: Enable the Denoising Semantic Refiner and logic-guided constraints for the final 20 epochs.
4. **Validation**: Use 5-fold cross-validation on CholecT45. Report $AP_I, AP_V, AP_T$ and the combined $AP_{IVT}$.

### 6. Expected Contribution

This model outperforms previous SOTA by (1) utilizing **Bipartite Binding** to ensure the instrument and target are spatially correlated before verb prediction, and (2) using **Generative Refinement** to enforce clinical logic, which significantly boosts the precision of tail-class predictions that are usually lost in the "noise" of common classes.