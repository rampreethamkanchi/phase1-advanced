# Research Project: Knowledge-Guided Relational Reasoning for Intraoperative Risk Prediction

## 1. Project Motivation & Clinical Significance
Laparoscopic Cholecystectomy (gallbladder removal) is one of the most common surgical procedures globally. However, despite its routine nature, intraoperative complications—most notably **Bile Duct Injury (BDI)**—remain a significant risk, often leading to severe long-term morbidity. 

Existing deep-learning models for surgical assistance primarily focus on "Perception": identifying instruments, organs, and basic actions (Triplets). While effective at describing "what is happening," these models lack "Clinical Significance"—the ability to understand *why* a certain scene is risky based on medical literature and safety rules.

**The Goal**: To develop a **Neuro-Symbolic Surgical Assistant** that bridges the gap between raw visual perception and clinical reasoning. By grounding live surgical scene graphs in a Knowledge Graph derived from surgical textbooks, the system can predict risks and provide explainable alerts based on established medical safety protocols.

---

## 2. Dataset Ecosystem
To build and evaluate the system, we utilize three distinct data domains:
- **CholecT45 / CholecT50**: These datasets are used for training the **Perception Layer**. CholecT45 provides labels for surgical action triplets `<Instrument, Verb, Target>`, while CholecT50 adds high-level **Surgical Phase** annotations (e.g., Gallbladder Dissection, Clipping/Cutting).
- **SSG-VQA**: The Surgical Scene Graph (SSG) dataset provides the foundation for our **Relational Transformer**. It contains per-frame graph structures where nodes are anatomy/instruments and edges represent spatial/functional relations.
- **Surgical Literature (PDFs)**: Textbooks (e.g., "The SAGES Manual on Advanced Laparoscopic Surgery") and "Rules of Safety" serves as the source for the **Knowledge Layer**, defining the "Rules of the Scene."

---

## 3. Perception Layer: Tri-Diff-Transformer (TDT)
The **Tri-Diff-Transformer (TDT)** is our novel architectural backbone designed to improve triplet detection through multi-stage relational grounding.

### 3.1 Architectural Strategy: Layer-by-Layer Detail
- **Visual Backbone**: **Swin-B (Base) Transformer**. Output features are spatial-temporal tokens of dimension $d_{model}=1024$. The backbone produces a grid of $12 \times 12$ spatial tokens (144 total) per frame.
- **Banded Causal Temporal Encoder**: 
  - **Structure**: A 3-layer Transformer Encoder with 8 attention heads and a feedforward dimension of 2048.
  - **Causal Constraint**: We implement a **Causal Positional Encoding** paired with a strict **Banded Causal Mask** (lower-triangular). This ensures that the representation of frame $t$ is computed using only $t$ and previous context, preventing "future-leakage" during live inference.
  - **Pooling**: Spatial tokens are average-pooled into a single temporal vector per frame before encoding, summarizing the scene state.
- **Decoupled Multi-Task Queries**:
  - We initialize 192 learnable queries (64 for $Q_I$, 64 for $Q_V$, and 64 for $Q_T$).
  - **Bipartite Binding Layer**: Instead of independent prediction, $Q_I$ (Instrument) and $Q_T$ (Target) queries undergo a 2-layer cross-attention exchange.
    - $Q_I$ attends to $Q_T$ to verify physical engagement.
    - $Q_T$ attends to $Q_I$ to ensure anatomy is being acted upon.
  - **Joint Semantic Decoding**: The $Q_V$ (Verb) queries attend to the **concatenated joint memory** of $Q_I$ and $Q_T$. This forces the model to predict verbs (e.g., "Clip") only when the physical evidence for the necessary tools ("Clipper") and targets ("Cystic Artery") is already established in the binding layer.

### 3.2 Denoising Semantic Refiner (DSR): Probabilistic Logic
The DSR acts as a discrete diffusion refiner that maps noisy visual logits into a clinically valid distribution $P(\text{Triplet} | \text{Vision})$.
- **Iteration**: A 3-step MLP iteration using **GELU** activations.
- **Clinical Soft-Masking**: In each step, we apply a cross-modal constraint using a precomputed **Clinical Prior Matrix** $P(V|I,T)$.
- **Einsum Formulation**: We calculate the "Expected Verb Mask" using the batch-wise outer product of instrument and target probabilities: 
  $$\text{Mask}_V = \sum_{i \in I} \sum_{t \in T} P(I_i) \cdot P(T_t) \cdot \text{Prior}(V | i, t)$$
- **Logit Correction**: This mask is added to the visual logits as a log-probability correction, effectively "denoising" invalid triplets (e.g., suppressing "Grasper-Clip" actions).

### 3.3 Tail-Boosted Contrastive Memory (TBCM)
To solve the extreme class imbalance (long-tail), we implement a **FIFO Memory Bank**.
- **Mechanism**: The bank stores the 1024-dimensional latent queries ($z$) of the rarest 15% of classes (Tail Classes).
- **Tail-Boosted Supervised Contrastive Loss (SupCon)**:
  - For every tail sample, we pull $N$ positive historical samples from the memory bank.
  - We use the **InfoNCE** objective, calculated in FP32 with **LogSumExp** to maintain numerical stability during extreme similarity events.
  - This forces the model to cluster rare activities into highly discriminative regions of the embedding space.

---

## 4. Advanced Perception Modules (Phase 2 & 3)
### 4.1 Surgical Phase Detection (Plan A)
Leveraging the temporal features from the TDT, we add a **Temporal Transformer Head** (4 layers). By freezing the backbone and training this head on **CholecT50**, we prove that "action-aware" features allow the model to recognize high-level phases (e.g., distinguishing "Phase 1: Trocar Insertion" from "Phase 2: Stretching") with significantly higher temporal IoU.

### 4.2 Enhanced Surgical Scene Graph (Relational Transformer)
For Phase 3, we implement a **Relational Transformer** that creates an explicit graph representation of the scene:
- **Node Enrichment**: 15 maximum nodes are initialized and "enriched" via cross-attention with the Swin-B spatial grid.
- **Edge Predictor**: Predicts 18 distinct relation classes (e.g., `within`, `near`, `touching`, `left_of`).
- **Energy Sub-head**: A dedicated binary classifier predicts the "Active Energy State" by pooling information from instrument nodes to detect high-frequency visual cues like smoke or arcing.

---

## 5. Knowledge Layer: Risk Knowledge Graph (RKG)
The Knowledge Layer is built using open-source Large Language Models (LLMs) like **Llama 3** or **Qwen 2.5**.

### 5.1 The Extraction Pipeline
- **RAG for Medical Rules**: We use Retrieval-Augmented Generation to feed specific surgical textbook passages into the LLM.
- **Schema Enforcement**: The LLM is prompted to output JSON triplets: `{"subject", "relation", "object", "risk", "explanation"}`.
- **Seed Knowledge Integration**: We anchor the graph with **Seed Knowledge** (e.g., the safety rules for the Calot's Triangle) to ensure the LLM remains grounded in surgical reality.

### 5.2 Storage and Ontology Mapping
The system maps heterogeneous clinical text to a standardized **Surgical Ontology**:
- **Instruments (6 classes)**: grasper, bipolar, hook, scissors, clipper, irrigator.
- **Anatomy (12 classes)**: liver, gallbladder, cystic_plate, cystic_duct, cystic_artery, etc.
- **Relations (18 classes)**: including functional actions (clip, cut) and spatial relations (near, touching).

---

## 6. Reasoning Layer: Neuro-Symbolic Logic
The **Reasoner** acts as the final decision head.

### 6.1 Semantic Graph Matching
- **Symbolic Matcher**: Performs a direct lookup in the RKG. If the scene graph contains a triplet $S-R-O$ that matches a "Critical" rule in the KG, an immediate alert is triggered.
- **Embedding Matcher**: Utilizes **SapBERT** to project both the scene relations and the KG relations into a shared medical semantic space. This allows the system to recognize that "Hook Cautery near Bile Duct" is semantically equivalent to "Hook active near Cystic Duct."

### 6.2 Risk Alerts and Explainability
When a risk is detected, the system generates a structured alert:
- **Alert Levels**: None, Low, Medium, High, Critical.
- **Clinical Suggestion**: Generated by the LLM (Llama 3) based on the "Explanation" field in the RKG. For example: *"⚠️ Risk: High. Hook active near Cystic Artery. Thermal spread may cause bleeding. Suggestion: Maintain 5mm safety buffer."*

---

## 7. Implementation & Hardware Details

### 7.1 Data Engineering & Preprocessing
- **Sliding Window Causal Sampling**: The dataset is sampled using a sliding window of size $T=8$. For a frame at index $t$, the input clip consists of $[t-7, t-6, \dots, t]$. At video boundaries, the first frame is padded/replicated to maintain window size without leaking future data.
- **Dataset Splits**: CholecT45 videos are split into **35 for training, 5 for validation, and 5 for testing** to ensure zero-overlap in anatomical variance between splits.
- **Caching**: A robust `.pkl` caching system stores the parsed indices and attributes, reducing dataset initialization time from minutes to milliseconds after the first run.
- **Augmentation Pipeline**:
  - **Training**: `RandomResizedCrop` ($448 \times 448$, scale 0.8-1.0), `RandomHorizontalFlip`, and `ColorJitter` (brightness/contrast 0.1).
  - **Inference**: Fixed `Resize` to $448 \times 448$ and standard Video-ImageNet normalization.

### 7.2 Training Strategy & Hyperparameters
- **Optimization**: **AdamW** optimizer with weight decay of 0.05.
- **Decoupled Learning Rates**: 
  - Backbone (Swin-B): $1e-5$
  - Task Heads (Temporal Encoder, Decoder, Refiner): $1e-4$
- **Loss Parameters**:
  - **Asymmetric Loss (ASL)**: Implements $\gamma_{neg}=4$ and $\gamma_{pos}=1$. This significantly down-weights the contribution of "easy negatives" (absent triplets), which constitute >99% of the label space.
  - **SupCon**: Temperature $\tau = 0.07$, Memory Bank size = 512.
- **Hardware Utilization**:
  - **Specs**: DGX Station A100 (80GB VRAM).
  - **Throughput**: `num_workers=32` on EPYC CPU ensures the GPU is never bottlenecked.
  - **Mixed Precision**: **Automatic Mixed Precision (AMP)** is enabled, with a gradient scaler to prevent underflow in the ASL log-calculations.
  - **Grad Accumulation**: Steps=2 to simulate larger batches (Effective Batch Size = 128).

---

## 8. Summary of Scientific Contributions
1. **Bipartite Binding**: The first surgical transformer to model instrument-target spatial coupling *before* action recognition.
2. **Generative Refinement (DSR)**: A novel use of discrete-diffusion-inspired iteration to enforce medical logic constraints.
3. **Neuro-Symbolic Risk Reasoning**: Bridging the semantic gap between SOTA vision models and textual medical literature through RKG construction and SapBERT alignment.
