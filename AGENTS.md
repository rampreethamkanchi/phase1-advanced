# Project Context: Knowledge-Guided Relational Reasoning for Intraoperative Risk Prediction

**CRITICAL**: Always run `conda activate neeshu` before running code.
We are doing this project for a research paper.

## 0. Logging & Training Standards
- **Comprehensive Industry-Standard Logging**: For all training, evaluation, and testing scripts, implement detailed logging that provides deep visibility into every process "behind the scenes."
- **Dual-Stream Output**: Logs must be written to both the terminal and a dedicated log file (e.g., `logs/run_<timestamp>.log`).
- **Insightful Observations**: Include processed interpretations of training dynamics within the log files, not just raw metrics. The goal is for the log to tell a detailed story of the run.
- **Debugging Readiness**: Ensure log files contain all necessary context for troubleshooting, as these will be used to diagnose failures.
- **Environment Constraints**: All scripts must operate within user-level permissions; no `sudo` or admin access is available.


## 1. Project Overview & Transition
**Goal**: Developing a **Neuro-Symbolic Surgical Assistant** that integrates deep-learning visual perception with symbolic reasoning for intraoperative risk prediction.
**Task**: Transition from visual facts (Phase 1 Triplets) to clinical reasoning by aligning live surgical scenes with Knowledge Graphs derived from medical literature.

**The Absolute Truth**: Neither me nor my mentor have experience in the medical field. We are relying on LLMs like you to bridge the gap from visual perception (Triplets) to clinical significance (Risk).
**Current Status**: Phase 1 (Triplet Detection) is complete. We are now moving into Phase 2 (Phase Detection) and Phase 3 (Enhanced Scene Graphs) leading to the Risk Reasoning module.
**The "Flavor" of the Project**:
- We implement two parallel paths: **Plan A (End-to-End Innovation)** and **Plan B (Integrative SOTA)**.
- Training scripts must support a `--plan [A/B]` flag to switch between these methodologies.
- The ultimate goal is to publish a research paper demonstrating the superiority of Knowledge-Guided reasoning over pure visual models.

## 2. Your Role: The Teacher & Collaborator
**This is critical**. I often feel lost when doing deep learning, specifically while writing code. **Not understanding makes me sick.**
- **You are my Teacher and Friend**: Treat me like a student. Explain concepts clearly.
- **Code Style**: **Heavily commented**. I need to understand *every bit* of the code. Do not just generic code; explain *why* we are doing it.
- **Incremental Steps**: Small steps at a time. Always create a "sample run" mode to test before full training.
- **Dataset Understanding**: You must deeply research CholecT50 and SSG-VQA. Dataset errors are the most common pitfall.

## 3. Next Objectives
- **Phase 2**: Surgical Phase Detection on **CholecT50**.
- **Phase 3**: Enhanced Surgical Scene Graph on **SSG-VQA** with "Attribute Enrichment".
- **Core Module**: Risk Knowledge Graph (RKG) Construction and Semantic Graph Matching.

## 4. Environment & Infrastructure
- **Specs**: DGX Station A100 (80GB VRAM).
- **GPU Usage**: Default training device is `cuda:0`.
- **LLM Inference**: Quantize Llama 3/Qwen to 4-bit or 8-bit to fit alongside vision models for "live" simulation.
- **Conda Environment**: `neeshu`
    - **CRITICAL**: Always run `conda activate neeshu` before running code.

## 5. Data Storage & Structure
All datasets are stored in `/raid` due to space constraints.
- **Root Dataset Dir**: `/raid/manoranjan/rampreetham/`
- **CholecT50 Location**: `/raid/manoranjan/rampreetham/CholecT50`
- **SSG-VQA Location**: `/raid/manoranjan/rampreetham/SSG-VQA`
- **Checkpoints**: Direct all outputs to `/raid/manoranjan/rampreetham/checkpoints`

## our plan for next phases

This implementation plan provides a comprehensive roadmap for developing a **Neuro-Symbolic Surgical Assistant**. The system integrates deep-learning-based visual perception (Surgical Scene Graphs) with symbolic reasoning (Knowledge Graphs derived from medical literature) to predict intraoperative risks.

Since **Phase 1 (Triplet Detection)** is complete as we together implemented, this plan focuses on the transition from visual facts to clinical reasoning.

So, you will implement both plan A and plan B, and while running, you will give me a flag in the command which will run either plan A or plan B. You got it right?

---

# Research Project: Knowledge-Guided Relational Reasoning for Intraoperative Risk Prediction

## 1. System Architecture Overview

The system consists of three distinct layers:

1. **Perception Layer (Visual):** Extracts "What is happening?" (Triplets, Phases, Enhanced Scene Graphs).
2. **Knowledge Layer (Symbolic):** Extracts "What is safe?" (Risk Knowledge Graph from medical PDFs).
3. **Reasoning Layer (Neuro-Symbolic):** Computes "Is the current scene risky?" by aligning the live scene with the ground-truth knowledge.

---

## 2. Plan A: The End-to-End Innovation Path

**Core Idea:** You build every sub-module from scratch or fine-tune extensively using your own architectures. Innovation lies in **Representation Transfer** from Phase 1.

### Phase 2: Surgical Phase Detection (CholecT50)

* **Methodology:** Leverage the feature representations from your custom Phase 1 Triplet Model.
* **Architecture:** Use the frozen (or lightly fine-tuned) backbone from the triplet model as a feature extractor. Attach a **Temporal Transformer** or a **Multi-Stage TCN (MS-TCN++)** head to capture long-range temporal dependencies between phases.
* **Innovation:** Prove that triplet-aware features (understanding tool-tissue interaction) accelerate and improve phase classification.

### Phase 3: Enhanced Surgical Scene Graph (SSG-VQA)

* **Methodology:** Use the SSG-VQA dataset as a baseline but implement "Attribute Enrichment" modules.
* **Enrichment Sub-tasks:**
* **Energy State Classifier:** A binary head to detect if tools (Hook/Clipper) are "active" (smoke/spark detection).
* **Proximity Head:** A regression layer predicting the distance (in pixels/normalized scale) between instrument tips and "forbidden zones" (e.g., Common Bile Duct).
* **Tissue Condition Detector:** A multi-label head for detecting bleeding, bile leakage, or excessive traction.


* **Innovation:** Develop a **Relational Transformer** that updates organ nodes based on instrument interactions (e.g., the gallbladder node changes state if the grasper is "pulling" it).

---

## 3. Plan B: The Integrative SOTA Path

**Core Idea:** Use established State-of-the-Art (SOTA) models for perception tasks to maximize data quality, shifting all research innovation to the **Knowledge Graph Reasoning**.

### Phase 2 & 3: Model Orchestration

* **Phase Detection:** Deploy a pre-trained SOTA model for CholecT50 (e.g., **TeCNO** or **TMR**).
* **Scene Graphs:** Implement the **SSG-VQA** baseline model or a SOTA graph generator like **Rendezvous** or **TriQuery**.
* **Plug-and-Play Integration:** Build a unified "Scene Buffer" that collects outputs from these separate models into a single, standardized JSON graph per frame.
* **Innovation Focus:** The "Research Contribution" here is not the vision model, but the **Fusion Logic**—how you reconcile conflicting outputs from different models to form a coherent surgical state.

---

## 4. The Core Research Module: Knowledge Graph & Risk Reasoning

*This module is required for both plans and represents the primary scientific contribution.*

### Step 1: Risk Knowledge Graph (RKG) Construction

* **Data Source:** Laparoscopic Cholecystectomy textbooks and "Rules of Safety" PDFs.
* **Engine:** Use open-source LLMs (**Llama 3 / Qwen 2.5**) via a local inference server (vLLM or Ollama).
* **Extraction Pipeline:**
1. **Structured Parsing:** Extract text using GROBID or PyMuPDF.
2. **RAG-based Extraction:** Use Retrieval-Augmented Generation to feed the LLM specific passages about "complications" and "critical view of safety."
3. **Ontology Mapping:** Force the LLM to output triplets in a schema that matches your SSG: `(Instrument) -> [Relation/Condition] -> (Anatomy) | [Risk Level] | [Intervention]`.
4. **Graph Storage:** Store in **NetworkX** (for research) or **Neo4j** (for production).



### Step 2: Semantic Graph Matching (The Reasoner)

* **Symbolic Matcher:** Compare the live SSG triplets against the RKG.
* *Example:* If SSG detects `(Hook) - [Active] - [Near] - (Bile Duct)` and RKG contains `(Electrocautery) - [Near] - (Bile Duct) -> [High Risk]`, trigger an alert.


* **Embedding Matcher:** Use a medical-domain embedding model (like **SapBERT**) to compute cosine similarity between scene relations and KG relations to handle synonyms (e.g., "Hook Cautery" vs. "L-Hook").
* **Reasoning Output:** Generate a natural language explanation using the LLM: *"Warning: Thermal spread detected near the Common Bile Duct. Literature suggest maintaining 5mm distance to avoid biliary stricture."*

---

## 5. Implementation Priorities for the Coding Agent

### Hardware Utilization (1x A100 80GB)

* **Data Loading:** Use `num_workers=32` on the EPYC CPU to ensure the GPU is never idle.
* **LLM Inference:** Quantize Llama 3/Qwen to 4-bit or 8-bit to fit alongside the vision models in VRAM for "live" simulation.
* **Storage:** Direct all checkpointing and dataset storage to `/raid/checkpoints` and `/raid/datasets`.

### Data Engineering

* **Standardization:** Create a canonical mapping for the Cholec family datasets. Ensure "Hook" in T45 is the same ID as "Hook" in SSG-VQA.
* **Weak Supervision:** Since "Risk Labels" are unavailable, use the RKG to auto-label (silver labels) your training data. If the KG says a state is risky, label that frame as "High Risk" to train a downstream binary risk-prediction head.

---

## 6. Evaluation & Publication Strategy

To publish a SOTA paper, the agent must implement:

* **Ablation Study:** Compare Risk Prediction accuracy with vs. without the Knowledge Graph (show that KG reduces false positives).
* **Metrics:** * **mAP** for triplets/SSG.
* **F1-Score** for risk alerts.
* **Time-to-Alert:** How many frames before a "complication" (like a bleed) the model predicted the risk.


* **Visualization:** Generate "Explainable Heatmaps" showing which node in the SSG triggered the KG match.

**Final Instruction for Agent:** "Follow Plan [A/B]. Prioritize the integration of Llama 3 for KG extraction and SapBERT for semantic alignment. All training must be optimized for a single 80GB A100."

## 6. Methodology: Plan A vs Plan B
You will implement both, switchable via a command-line flag.

### Plan A: The End-to-End Innovation Path
* **Phase 2**: Use frozen Phase 1 backbones with a **Temporal Transformer** or **MS-TCN++**.
* **Phase 3**: "Attribute Enrichment" (Energy State, Proximity, Tissue Condition) and a **Relational Transformer**.
* **Innovation**: Demonstrate that triplet-aware features enhance complex understanding.

### Plan B: The Integrative SOTA Path
* **Phase 2**: Deploy pre-trained SOTA (e.g., **TeCNO** or **TMR**).
* **Phase 3**: Implement **SSG-VQA** baseline (e.g., **Rendezvous** or **TriQuery**).
* **Innovation**: Focus on the **Fusion Logic** and the integration of diverse perception modules.

## 7. Core Research: Risk Knowledge Graph (RKG) & Reasoning
*This is the primary scientific contribution for both plans.*
- **Construction**: Use Llama 3 / Qwen 2.5 via local inference (vLLM/Ollama) to extract rules from textbooks.
- **Pipeline**: Structured parsing (GROBID) -> RAG extraction -> Ontology Mapping -> NetworkX/Neo4j.
- **Reasoner**: 
    - **Symbolic Matcher**: Logic-based alignment of SSG triplets with RKG rules.
    - **Embedding Matcher**: Use **SapBERT** for semantic similarity between scene nodes and KG nodes.
    - **Explainability**: Generate NL explanations: *"Warning: Thermal spread near Bile Duct. Minimum safe distance: 5mm."*

## 8. Evaluation & Publication Strategy
- **Ablation**: Compare Risk Prediction with vs. without Knowledge Graph.
- **Metrics**: mAP (Triplets/SSG), F1-Score (Risk Alerts), Time-to-Alert.
- **Visualization**: "Explainable Heatmaps" linking SSG nodes to KG triggers.

Remember: I accept you as my teacher and buddy. Let's build this from scratch, understanding everything. Take your own decisions for the best effect for our project, WE ARE PUBLISHING A RESEARCH PAPER.

If you want deeper explanation you read completely: [CHATGPT-plan.md](./CHATGPT-plan.md)
