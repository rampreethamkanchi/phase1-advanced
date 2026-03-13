# KGR-Surg: Knowledge-Guided Relational Reasoning for Safety-Aware Risk Prediction in Laparoscopic Surgical Videos

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)

## 🚀 Overview

**KGR-Surg** is a state-of-the-art framework that bridges the gap between low-level visual perception and high-level clinical reasoning in laparoscopic surgery. By combining deep-learning-based visual understanding with symbolic logic derived from medical literature, KGR-Surg predicts intraoperative risks with high precision and explainability.

### Key Contributions:
1. **Tri-Diff-Transformer (TDT)**: A novel architecture for instrument-verb-target triplet detection.
2. **Denoising Semantic Refiner (DSR)**: A symbolic reasoning module that enforces anatomical and clinical constraints on visual predictions.
3. **Risk Knowledge Graph (RKG)**: A structured knowledge base extracted from surgical textbooks using LLMs (Llama 3/Qwen) to guide risk reasoning.
4. **Enhanced Scene Graphs**: Explicit modeling of instrument-tissue proximities and energy activation states.

---

## 🛠️ System Architecture

The project is structured into four sequential phases:
*   **Phase 1 & 2: Structural Perception**: Detecting "Who is doing What to Whom" (Triplets) and identifying the current surgical phase (Phase Detection).
*   **Phase 3: Enhanced Scene Graphs**: Building a topological map of the surgical field, including instrument-organ relationships and energy activation states.
*   **Phase 4: Risk Reasoning**: Aligning the live surgical scene with the RKG to trigger critical risk alerts with natural language explanations.

---

## 📦 Installation & Environment

### Prerequisites
- NVIDIA GPU with at least 15GB VRAM (24GB+ recommended, optimized for A100 80GB).
- CUDA 11.8+

### Setup
We recommend using a Conda environment for isolation.

```bash
# Create and activate the environment
conda create -n kgrsurg python=3.10
conda activate kgrsurg

# Install dependencies
pip install -r requirements.txt
```

---

## 📂 Dataset Preparation

The system utilizes two primary benchmarks. Ensure your directory structure follows this pattern:

### 1. CholecT50 (Phases 1 & 2)
Available from the [Surgical Action Triplet](https://github.com/Rendia/CholecT50) repository.
```text
/raid/manoranjan/rampreetham/CholecT50/
    ├── videos/           # Original video frames
    └── labels/           # Triplet and Phase annotations
```

### 2. SSG-VQA (Phase 3)
Available from the [SSG-VQA](https://github.com/surgical-vision/SSG-VQA) repository.
```text
/raid/manoranjan/rampreetham/SSG-VQA/
    ├── images/           # Frame images
    └── scene_graphs/     # Ground truth JSON graphs
```

---

## 🏃 Run Instructions

### Phase 1 & 2: Triplet and Phase Training
Train the perception backbone on CholecT50 using Plan A (Temporal Transformer).

```bash
python src/train.py \
    --dataset_dir /raid/manoranjan/rampreetham/CholecT50 \
    --dataset_type cholecT50 \
    --plan A \
    --batch_size 64 \
    --epochs 40
```
*Use `--sample_run` for a quick 2-epoch verification.*

### Phase 3: Surgical Scene Graph Training
Train the relational transformer with frozen perception features.

```bash
python src/train_ssg.py \
    --dataset_dir /raid/manoranjan/rampreetham/SSG-VQA \
    --pretrained_backbone logs/best_phase2_model.pt \
    --plan A \
    --batch_size 16 \
    --epochs 20
```

### Phase 4: Risk Knowledge Graph (RKG) Construction
Extract rules from medical literature using local LLM inference (requires Ollama/vLLM).

```bash
# Standard extraction (requires Llama 3 running on Ollama)
python -m src.rkg.extract_knowledge --pdf_dir /path/to/pdfs --model llama3

# Mock extraction (Load pre-defined seed knowledge for quick setup)
python -m src.rkg.extract_knowledge --mock
```

---

## 📊 Evaluation & Metrics

We provide a unified evaluation script that computes all metrics reported in the paper, including **Critical Risk Recall (CRR)**, **Anatomical Impossible Rate (AIR)**, and **Risk-Weighted mAP**.

### Standard Evaluation
```bash
python src/evaluate_all_phases.py \
    --t50_dir /raid/manoranjan/rampreetham/CholecT50 \
    --ssg_dir /raid/manoranjan/rampreetham/SSG-VQA
```

### Fast Evaluation (Memory Optimized)
Optimized for submission deadlines or lower VRAM environments.
```bash
python src/fast_eval_for_submission.py --less_mem
```

---

## 🧪 Demo: Risk Reasoning
Experience the neuro-symbolic reasoner in action with a sample scene:
```bash
python src/reasoner_demo.py
```

---

## 📜 Citation

If you find this work useful in your research, please cite:
```bibtex
@article{kgrsurg2026,
  title={KGR-Surg: Knowledge-Guided Relational Reasoning for Safety-Aware Risk Prediction in Laparoscopic Surgical Videos},
  author={Rampreetham Kanchi, et al.},
  journal={ECML PKDD},
  year={2026}
}
```

## ⚖️ License
This project is licensed under the MIT License - see the LICENSE file for details.
