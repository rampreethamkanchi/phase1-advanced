Great — I understand. You finished **phase 1 (triplet detection)** yourself, so I won’t touch that. Below is a single, clear, final plan for the **remaining work** (phase 2 onward), in **two variants** you requested:

* **Plan A (Full research path)** — we build our own models for every phase (phase detection, scene-graph enrichment, risk KG reasoning). This gives maximal novelty and more paper-worthy contributions.
* **Plan B (Practical / engineering path)** — we reuse strong open-source models for all perception tasks, and focus innovation on the **knowledge graph (KG)** and the risk reasoning pipeline. Faster, cheaper, still publishable if KG + reasoning is novel.

I keep language simple, give examples, and show concrete steps the coding agent can implement. No code here — this is the instruction spec you’ll paste to the coding agent.

Key full-forms (first time): LLM (large language model), KG (knowledge graph), SSG (surgical scene graph), RAG (retrieval-augmented generation), GNN (graph neural network), OCR (optical character recognition), CVAT (Computer Vision Annotation Tool), FAISS (Facebook AI Similarity Search).

---

# Assumptions & inputs (what you already have)

* **Phase 1 (triplet detection)**: done — you have a trained model and outputs (triplets per frame/time). Example: at t=45s you have `<Hook, Dissect, Cystic Duct>` with scores.
* Datasets available: **CholecT45/CholecT50/CholecT family**, **SSG-VQA**, CAMMA family (see [https://camma.unistra.fr/datasets/](https://camma.unistra.fr/datasets/)), and many PDFs of surgical literature (textbooks, PubMed Central).
* You want **open-source** LLMs only (Llama family or Qwen family). Good — will use them for text-to-KG extraction and KG reasoning.
* Final goal: a research paper titled similar to your abstract: *Knowledge-Guided Relational Reasoning for Video-Based Intraoperative Risk Prediction in Laparoscopic Cholecystectomy*.

---

# Outputs we will produce (what your coding agent should deliver)

1. **Phase-2 model** (phase detection) — model + checkpoints + per-frame phase labels.
2. **Enhanced SSG pipeline** — SSGs with extra node/edge attributes (energy state, tissue state, landmark flags, instrument velocity, occlusion flags).
3. **Knowledge Graph (KG)** built from surgical literature PDFs — normalized ontology and provenance.
4. **Risk-matching engine** — compares live/enriched SSG with KG and produces risk predictions (probabilistic scores + natural language suggestions).
5. **Evaluation suite** — metrics, ablation scripts, visualization tools (attention maps, graph views).
6. **Paper-ready experiments & figures** — tables for mAP/AP, head-vs-tail, risk detection precision/recall, time-to-detect curves, sample inference snapshots.

---

# Short reasoning summary (why this plan)

* Triplets give *who does what to what*. But risk needs *context*: phase, spatial relations, instrument energy, tissue condition, motion. So we must **enrich SSG** and then **ground** those enriched relations to a KG extracted from literature. Using open LLMs + RAG + KG + graph reasoning gives a neuro-symbolic system — good novelty for a paper. Example: if SSG shows `Hook active` + `near` + `Cystic Duct` + `phase != clipping`, KG can say “High-risk: thermal injury to bile duct” and suggest “stop cautery, increase distance”.

---

# Common components shared by Plan A & Plan B

These modules are common; implement once and reuse:

1. **Data manager**: standardized JSON schema for frames/clips, SSG nodes/edges, triplets, phases, and extra attributes. Example JSON for one frame:

```json
{
  "frame_id": "...",
  "phase": "calot_dissection",
  "triplets": [{"instrument":"hook","verb":"dissect","target":"cystic_duct","score":0.84}],
  "ssg": {"nodes":[...],"edges":[...]},
  "extra_attrs": {"hook_active": true, "hook_distance_mm": 3.2, "hook_velocity_mm_s": 25}
}
```

2. **Ontology**: canonical vocabularies for instruments, verbs, anatomical structures, complications. Map synonyms to canonical forms (e.g., `L-hook`, `electrocautery` → `hook_cautery`). Example mapping stored in YAML.

3. **KG store**: use **Neo4j** (community edition) for production-like queries, and **NetworkX** for prototyping and experiments. Store node types, relations, provenance (paper id + sentence), and a numeric `risk_score` for relations derived from literature.

4. **RAG pipeline**: vector search over literature passages (FAISS) + LLM to extract and verify relations, using Llama/Qwen as the LLM and a sentence-transformer for embeddings. Example embedding model: `sentence-transformers/all-mpnet-base-v2` (open-source).

5. **Evaluation datasets**: define risk-labeled test set (manual or weak-supervised). Metrics: precision, recall, F1, average precision (AP), time-to-detection, calibration (ECE).

---

# Phase 2: Phase detection (changed dataset: CholecT50)

(You said phase 2 uses CholecT50 now.)

Goal: get reliable **phase** labels per-frame or per-clip. Phase helps disambiguate allowable actions (e.g., clipping allowed in clipping phase).

## Plan A (build your own phase detection model)

* **Input**: frames/clips from CholecT50; you already have a backbone from phase1 (transfer representations).
* **Model**:

  * Use the phase representations from your triplet model as warm-start features (you said you can reuse). Example: take encoder features (spatial–temporal) from triplet model and add a small temporal classifier head.
  * Temporal model options: Temporal Transformer or Bi-LSTM over clip features. My opinion: Temporal Transformer (4 layers) with position encodings — more flexible.
  * Output: softmax over phases + confidence.
* **Training**:

  * Loss: cross-entropy + label smoothing.
  * Use class-balancing if some phases under-represented.
  * Data augmentations: same as triplet.
* **Evaluation**:

  * Per-phase accuracy, per-frame F1, segment-level IoU (temporal).
* **Example**: reuse feature extractor weights, add `phase_head.py` that takes `[B, T, D] → [B, num_phases]`.

## Plan B (reuse SOTA)

* Use a pre-trained surgical phase detection model if available (open-source): e.g., models that won M2CAI challenge or recent transformer-based implementations in GitHub. Fine-tune on CholecT50.
* Advantage: less training time; only adapt to domain shift.

---

# Phase 3: Surgical Scene Graph (SSG) enrichment — deriving more features

You already have SSG-VQA source. But for risk we need extra attributes. This is critical.

## Extra signals to extract (and how to get them)

1. **Instrument energy state (active/inactive)**

   * Method: classify instrument tip region as active (sparking/smoke/visible cautery glow) vs inactive. Use a small CNN classifier on crops around tool tip. Train with annotated frames where hook is active. Example: use YOLOv8 to detect tool tip then a ResNet18 classification head.

2. **Tissue states**: bleeding, bile leak, inflammation, thermal damage

   * Method: segmentation + patch classification. Use Mask R-CNN / Segmenter (SAM for proposal + classifier). Train on small manually-annotated set or weak-label from color heuristics (bleeding = red region + texture change). Example: bleeding detection uses red-color + optical flow presence.

3. **Anatomical landmarks for CVS (critical view of safety)**

   * Detect cystic duct, cystic artery, Rouviere’s sulcus, hepatocystic triangle clear. Use a dedicated detector trained on annotated samples (manual or CAMMA labels if exist). Example: output boolean `CVS_established`.

4. **Instrument tip distance to structure (metric distance)**

   * Method: use detected bounding boxes + homography-based distance approximation or use size of instrument tip pixels to approximate mm. Example: calibrate from known instrument length at start frame, or use stereo if available; if not, use relative pixels and thresholds (e.g., `near` if < 10 px normalized).

5. **Temporal dynamics (velocity / sudden motion / force proxy)**

   * Method: track instrument tip across frames (Kalman filter / Deep SORT) and compute velocity. Sudden acceleration flag = potential risk. Example: `velocity_mm_s > threshold`.

6. **Out-of-view instrument tracking**

   * If instrument bounding box disappears but was present a moment ago and the shaft indicates insertion, mark `tip_out_of_view = true`. This requires tracking and heuristics.

7. **Occlusion / poor view / lens fog**

   * Method: drop in confident detections + blur detection → penalize confidence and raise "uncertain" risk.

8. **Relation types richer than `near`**

   * Add relations: `touching`, `cutting`, `grasping`, `pulling`, `pressing`, `thermal_spread`. Use classifier per pair of nodes (visual + motion cues).

## Implementation choices

* Use **Detectron2** or **Ultralytics YOLOv8/YOLOv9** for detection & segmentation (open-source). Use **SAM (Segment Anything Model)** for fast annotation/seeding and then fine-tune downstream heads.
* Use optical-flow (Farneback or RAFT) to derive motion fields for velocity; RAFT (open-source) is good.
* Use **DINOv2** or ViT features for few-shot recognition of landmarks.
* Example pipeline per frame:

  1. Detect instruments & organs (detect model).
  2. Segment organ regions (segment model).
  3. Compute instrument-tip crop → energy classifier.
  4. Track tips across frames → compute velocity.
  5. Compute pairwise relations using spatial overlap + relative motion → relation classifier.

## How to get missing labels (annotation strategies)

* **Semi-automatic labeling**: use SAM + a small set of manual masks to propagate.
* **Synthetic augmentation**: copy-paste instrument tips, or simulate smoke for active hook to enlarge dataset.
* **Weak supervision**: use rule-based heuristics to auto-label (`red-color & flow` → bleeding).
* **Crowd or expert annotation** in CVAT for critical landmarks.

---

# Phase 4: Knowledge Graph (KG) from literature — extraction & normalization

You will use open-source LLMs (Llama family or Qwen) and embeddings for RAG. This module is core innovation for Plan B, and part of Plan A as well.

## KG building pipeline (high-level)

1. **PDF ingestion + OCR**

   * Use `GROBID` (open-source) for structured parsing of papers and `PyMuPDF`/`pdfplumber` as fallback. For scanned images use Tesseract OCR.
   * Extract sections, captions, and references.

2. **Chunking + Indexing**

   * Chunk text into logical paragraphs (2–8 sentences), keep provenance metadata (paper id, section, page, sentence index).
   * Create vector index (FAISS) of chunks using sentence-transformer embeddings (e.g., `all-mpnet-base-v2`) for retrieval.

3. **LLM extraction** (open-source Llama / Qwen)

   * Use a RAG setup: retrieve top-k relevant chunks for a query (e.g., “risks from Hook near Cystic Duct”) and ask the LLM to **extract structured triplets**: `(instrument/action) — relation — (anatomy) => (complication); risk_level; evidence_sentence`.
   * Use templates & few-shot examples in the prompt for stable output.
   * Use **LoRA (low-rank adaptation)** fine-tuning if large domain adaptation needed (fine-tune Llama2 or Qwen with small in-domain dataset).

4. **Entity normalization**

   * Map extracted strings to canonical ontology (use synonym lists or embedding similarity to canonical nodes).
   * Example: "electrocautery stick" → canonical `hook_cautery`.

5. **KG storage**

   * Create nodes: `Instrument_Action`, `Anatomy`, `Complication`, `Phase`, `Condition` (e.g., `adhesion`).
   * Create edges: `causes_risk_of`, `near_to`, `unsafe_in_phase`, `increases_prob_of`, with attributes: `risk_level` (High/Med/Low or numeric), `evidence` (paper id + sentence), `confidence`.
   * Store in Neo4j; also export readable NetworkX graph for experiments.

6. **Ranking & scoring**

   * For each KG relation, compute a numeric `prior_risk_score` from evidence count + source trust + LLM confidence (simple weighted sum). Example: `prior = 0.6 * evidence_count_norm + 0.4 * LLM_conf`.

7. **Small example** (conceptual):

   * From literature: “Thermal spread from electrocautery near common bile duct causes biliary injury.” → node: `hook_active` → edge `thermal_spread_near` → `common_bile_duct` with `risk_level=High`, `complication=Biliary_injury`, `evidence=paperX:pg12`.

## Tools & models (all open-source)

* PDF parsing: `GROBID`, `pdfplumber`, `PyMuPDF`.
* OCR: `Tesseract` if needed.
* Embeddings: `sentence-transformers/all-mpnet-base-v2`.
* Vector DB: `FAISS`.
* LLMs: **LLaMA 2/3** (if license OK) or **Qwen-1.5/2** open-source variants. Use local inference (llama.cpp or vLLM or Hugging Face runner).
* Adapter: **LoRA** for efficient fine-tuning.
* KG DB: **Neo4j** community or **NetworkX** for offline.

---

# Risk matching: how to detect risk from SSG + KG

We need a deterministic + learned hybrid:

## Step A — Symbolic matching (fast, explainable)

* For each SSG entry (instrument, relation, target, extra attributes like `active`, `distance`, `phase`), query KG for edges that match instrument→relation→anatomy triplet.
* If there is a matching edge with `risk_level` and the scene attributes meet thresholds (e.g., `distance < d_threshold`, `active==True`, `phase != safe_phase`), then **raise an alert** with that KG edge’s risk score.
* Example rule: if KG has `hook_active — thermal_spread_near — common_bile_duct` with risk_level=High and SSG shows `hook_active=true`, `distance_mm < 5` → predict `High` risk.

## Step B — Graph similarity + probabilistic reasoning (robust)

* Convert both **enriched SSG** and **relevant subgraph(s)** from KG into graph embeddings (use GNN or GraphSAGE). Then compute embedding similarity — this provides fuzzy match even if exact labels differ.
* Use a learned logistic regressor or small MLP that takes features:

  * match_score (graph similarity),
  * prior_risk_score (from KG),
  * visual_confidences (triplet model score, instrument detection score),
  * temporal flags (velocity, repeated proximity),
  * phase_confidence.
  * Output: probability of risk and suggested action.
* Train this classifier on small labeled set (manual) or weak supervision (derive labels from literature + human curation).

## Step C — Temporal aggregation & early warning

* Aggregate frame-level risk probabilities into temporal windows to reduce false positives. Example: raise emergency alert if `prob_risk >= 0.85` for `>= 3` consecutive frames or `prob_risk >= 0.95` for single frame.
* Also track `time-to-detection` metric: earlier detection == better.

## Explainability

* For each alert, show:

  * matched KG rule(s) with evidence sentences,
  * contributing visual features and their confidences,
  * suggested action in plain language (e.g., “Stop cautery; retract instrument; verify CVS”).
* Example output (structured + NL):

```json
{
  "time": 123.5,
  "risk": "High",
  "reason": "Hook active within 3mm of common bile duct; literature reports thermal spread causes duct injury (paper id X).",
  "suggestion": "Stop cautery, move instrument away, confirm CVS."
}
```

---

# Plan A (build our own models end-to-end) — detailed steps

This plan yields highest novelty because you will design new models, transfer representations from phase1, and integrate KG tightly.

## Phase A1 — Reuse & adapt triplet encoder

* Reuse the triplet encoder backbone (since you built it).
* Freeze early layers initially; fine-tune heads for phase detection and SSG attribute heads.
* Example: encoder outputs shared `E(x)`; attach heads:

  * `phase_head(E)`, `energy_head(E, crop_tip)`, `tissue_head(E, crop_patch)`, `relation_head(E, pair_features)`.

## Phase A2 — Train SSG attribute detectors (from scratch)

* Create labeled set for instrument energy, tissue states, and landmarks (mix manual + weak labels).
* Train detectors sequentially: detection → segmentation/pose → attribute classifiers.

## Phase A3 — Relation prediction network

* Build a pairwise relation classifier that ingests two node features + relative geometry + motion features. Use a small MLP or a relation transformer.
* Output relation type + confidence.

## Phase A4 — Graph enrichment & tracker

* Implement instrument tip tracker (Kalman/Optical flow + detection).
* Compute velocities and occlusion flags.
* Attach all attributes to SSG nodes/edges.

## Phase A5 — KG extraction & KG alignment

* Build the KG as above using open LLMs; normalize entities.
* Implement symbolic matching rules and the learned fusion classifier (GNN + MLP).

## Phase A6 — Training risk predictor

* If you have labels: supervised training of risk predictor. If not, use weak supervision:
* Here we do not have labels for risk, so we will use weak supervision:
  * Generate silver labels by pattern-matching literature rules,
  * Human-verify a small set,
  * Train MLP.
* Evaluate on held-out set, cross-validation.

## Phase A7 — Ablations & experiments

* Ablate: no energy state, no velocity, no KG, replace learned fusion with symbolic only, etc.
* Compare per-risk AP and early-warning time.

## Advantages (opinion)

* Full control, highest chance of publishing a novel model combining perception + KG + reasoning.
* You can claim novel representation transfer (triplet→phase→KG).

## Disadvantages

* Big annotation burden, more compute, longer timeline.

---

# Plan B (reuse SOTA perception, focus novelty on KG & reasoning) — detailed steps

This plan is practical and faster. Scientific novelty focuses on KG construction, KG-grounded reasoning, and evaluation.

## Perception (plug-and-play)

* Use open-source pre-trained models:

  * **Instrument & organ detection**: YOLOv8 or Detectron2 pre-trained on Cholec datasets.
  * **Segmentation / SAM**: Segment Anything Model to seed masks.
  * **Energy-state**: small classifier trained from few-shot examples (could be fine-tuned).
  * **Phase detection**: use an off-the-shelf surgical-phase detection repo (M2CAI winners).
  * **Tracking**: Deep SORT or ByteTrack variant.
* Fine-tune minimally on your data (CholecT50) to adapt to appearance.

## KG & reasoning (focus area)

* Implement full KG pipeline (ingest literature, RAG, LLM extraction, normalization).
* Implement matching pipeline (symbolic + learned fusion).
* Build UI / visualization to show matched evidence for every alert.

## Advantages (opinion)

* Much faster to build a working system and run experiments.
* Strong focus on KG novelty: extraction, uncertainty scoring, graph-based reasoning, evidence-grounded suggestions. Good for a solid paper if KG reasoning is novel and well-evaluated.

## Disadvantages

* Less novelty on perception side; reviewers may expect some new perception contributions — but KG novelty + clinical evaluation can be strong.

---

# Data augmentation & synthetic data suggestions (both plans)

* **Instrument tip copy-paste**: create more `hook_active` examples by copying sparks/smoke patches.
* **Simulate thermal spread**: add colored halos to instrument-tip to mimic thermal glow for classifier training.
* **Domain randomization**: change brightness/contrast for robustness.
* **Weak label propagation**: use high-confidence model predictions to auto-annotate more frames.

Example: to enlarge `cystic_artery` landmark labels, manually label 200 frames, then use a detector to pseudo-label 2000 more.

---

# Ground truth risk labels (how to create)

* **Manual expert labels** (best): ask surgeons to mark risk moments. Even small number (200–500 clips) is valuable.
* **Weak supervision**: rules from KG can generate silver labels (if instrument=hook_active & distance small & phase != clipping → label risk=True).
* **Data programming**: use Snorkel-style rule aggregation to reconcile weak labels, then train risk model.

---

# Evaluation metrics (what to report in paper)

* **Detection metrics**: mAP for triplet detection (phase 1 already done).
* **Phase detection**: accuracy, F1, temporal IoU.
* **SSG enrichment**: per-attribute AP (energy-state AP, bleeding AP).
* **Risk detection**:

  * Precision, Recall, F1 (per risk class).
  * Average Precision (AP) across risk types.
  * Time-to-detect (median seconds prior to event).
  * False Alarm Rate (per minute).
  * Calibration (ECE).
* **Ablation tables**: show effect of KG, energy state, velocity, and phase on risk AP.
* **Qualitative**: case studies, attention maps, KG evidence snippets.

---

# Experiments & ablation plan (must include)

1. Full system vs symbolic-only vs perception-only.
2. Plan A vs Plan B (end-to-end vs plug-and-play).
3. Removing each enriched attribute (energy, velocity, landmark) one at a time.
4. KG variants: raw KG priors vs normalized KG vs KG + learned risk score.
5. Early-warning threshold sweep (precision/recall curve with different temporal aggregation).
6. Human-in-the-loop test: show alerts to surgeons and collect feedback.

---

# Deliverables & file structure (what to give the coding agent)

* `data/` — split files, ontology, canonical maps.
* `src/perception/` — detectors, trackers, attribute heads (Plan A) or wrappers to SOTA models (Plan B).
* `src/kg/` — ingestion, RAG, LLM prompt templates, extraction + normalization.
* `src/reasoning/` — symbolic matcher, graph-embedding matcher, fusion model training.
* `src/eval/` — metrics, plotting, ablation runner.
* `notebooks/` — visual analysis and sample inference.
* `docs/` — KG schema, ontology, README.
* `results/` — per-experiment metrics and plots.

---

# Compute & timeline (opinionated estimate)

* **Plan A (full)**: 3–6 months (annotation + training + KG + evaluation) on 2–4 GPUs (A100 or 3090s). More if annotation slow.
* **Plan B (practical)**: 4–8 weeks to get a working prototype + KG reasoning with strong results; 2–3 months for thorough experiments and paper.
* If you need to save cost, choose Plan B first; then do Plan A components selectively for novelty.

---

# Risks & mitigations (practical)

* **No ground-truth risk labels** — mitigations: weak supervision + small expert annotation + careful evaluation.
* **LLM hallucination when extracting KG** — mitigations: always attach provenance, weight edges by evidence count, and verify extractions using multiple LLMs or consensus.
* **Distance/scale ambiguity in laparoscopy** — mitigations: define relative distance thresholds using instrument pixel sizes and validate with surgeons.
* **Class imbalance (rare risk events)** — mitigations: synthetic augmentation, oversampling, focal loss, contrastive learning for rare classes (idea you already used in triplets).

---

# Paper outline + contributions (to include in the manuscript)

1. **Introduction** — define problem, clinical importance.
2. **Related work** — triplet detection, scene graphs, surgical KG, RAG.
3. **Method**

   * Perception & SSG enrichment (explain new attributes).
   * KG construction from literature (RAG + LLM + normalization).
   * Risk matching: symbolic rules + learning-based fusion.
4. **Experiments**

   * Datasets, annotation, evaluation metrics.
   * Quantitative results + ablations.
   * Qualitative case studies + explanation examples.
5. **Discussion** — limitations, clinical utility, human-in-loop.
6. **Conclusion** — summary + future directions.

**Novel contributions to highlight**:

* Enriched SSG attributes tuned for risk prediction (energy, velocity, CVS flags).
* Open-source KG of surgical risks with provenance.
* Neuro-symbolic matching pipeline combining KG priors with learned fusion for early risk prediction.
* Thorough ablation showing KG improves recall with fewer false alarms.

---

# Final recommendations (opinionated)

* If you want a **publishable** novel system with manageable work: start with **Plan B** (reuse SOTA perception) and pour research effort into KG building + reasoning + rigorous evaluation with surgeons. This will give faster results and clear novelty.
* If you want **maximum novelty** and can invest time and annotation: do **Plan A** and reuse triplet encoder as a transfer backbone — that’s more likely to be "best in class" but needs more work.
* Always attach **provenance** to KG edges and keep LLM-driven outputs verifiable. Example: in paper, show the exact sentences used as evidence for high-risk edges.

---
