# Methodology Context: Phase 4 Risk Reasoning

This document contains the complete source code for all modules contributing to this phase. Use this as context for writing the methodology section of the research paper.

## File: `src/rkg/extract_knowledge.py`

```python
import os
import json
import argparse
import requests
import fitz  # PyMuPDF
from .ontology import SURGICAL_ONTOLOGY, SEED_KNOWLEDGE

class KnowledgeExtractor:
    """
    Parses Medical PDFs and uses LLM (Ollama) to extract (Subject, Relation, Object) risk triplets.
    """
    def __init__(self, ollama_url="http://localhost:11434/api/generate", model="llama3"):
        self.ollama_url = ollama_url
        self.model = model
        
    def extract_text_from_pdf(self, pdf_path):
        """Extracts raw text from a PDF file."""
        text = ""
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                text += page.get_text()
            doc.close()
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}")
        return text

    def get_triplets_from_llm(self, text_chunk):
        """Calls local Ollama to extract high-impact surgical risk triplets."""
        prompt = f"""
        Analyze the following surgical text and extract Critical Risk Rules for Laparoscopic Cholecystectomy.
        
        FOCUS: 
        1. "Critical View of Safety" (CVS) requirements.
        2. Anatomical danger zones (e.g., Common Bile Duct, Hepatic Artery).
        3. Instrument-tissue interaction risks (e.g., thermal spread from L-hook, improper clipping).
        
        ABSOLUTELY CRITICAL: You MUST use EXACTLY the words from these lists for your subjects, objects, and relations. DO NOT INVENT NEW WORDS.
        ALLOWED SUBJECTS AND OBJECTS: {SURGICAL_ONTOLOGY['instruments'] + SURGICAL_ONTOLOGY['anatomy']}
        ALLOWED RELATIONS: {SURGICAL_ONTOLOGY['relations'] + SURGICAL_ONTOLOGY['spatial_relations']}
        
        If a text describes a risk that does not fit perfectly into these allowed words, DO NOT extract it. Skip it entirely.
        
        CRITICAL: Respond ONLY with a valid JSON list. 
        Format:
        [
          {{"subject": "<must be exactly an allowed word>", "relation": "<must be exactly an allowed relation>", "object": "<must be exactly an allowed word>", "risk": "Low/Medium/High/Critical", "explanation": "Detailed clinical reasoning why this is risky"}}
        ]
        
        Text:
        {text_chunk[:4000]} 
        """
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_ctx": 8192
            }
        }
        
        try:
            response = requests.post(self.ollama_url, json=payload, timeout=120)
            if response.status_code == 200:
                result = response.json()
                clean_response = result['response'].strip()
                # Simple extraction of JSON block if preamble exists
                if "[" in clean_response and "]" in clean_response:
                    start = clean_response.find("[")
                    end = clean_response.rfind("]") + 1
                    json_str = clean_response[start:end]
                    try:
                        return json.loads(json_str)
                    except:
                        # Fallback for minor formatting errors
                        return []
                return json.loads(clean_response)
        except Exception as e:
            print(f"LLM Error: {e}")
            return []
        return []

    def mock_extraction(self):
        """Returns seed knowledge as a mock result."""
        print("Using MOCK extraction mode...")
        return SEED_KNOWLEDGE

def merge_rules(existing_rules, new_rules):
    """
    Deduplicates rules based on (subject, relation, object, risk).
    Priority Logic:
    1. If new triplet doesn't exist: Add it.
    2. If exists with no explanation, but new one has one: Upgrade it.
    3. If already has an explanation: Keep the first one and discard others.
    """
    rule_map = {}
    
    valid_subjects_objects = [s.lower() for s in SURGICAL_ONTOLOGY['instruments'] + SURGICAL_ONTOLOGY['anatomy']]
    valid_relations = [r.lower() for r in SURGICAL_ONTOLOGY['relations'] + SURGICAL_ONTOLOGY['spatial_relations']]
    
    # Pre-populate map with existing rules
    for r in existing_rules:
        key = (r['subject'].lower(), r['relation'].lower(), r['object'].lower(), r['risk'].lower())
        rule_map[key] = r
        
    for nr in new_rules:
        # Basic validation
        if not all(k in nr for k in ['subject', 'relation', 'object', 'risk']):
            continue
            
        subj = str(nr['subject']).lower().strip()
        rel = str(nr['relation']).lower().strip()
        obj = str(nr['object']).lower().strip()
        
        # VERY IMPORTANT: Filter out hallucinated vocabulary (e.g. "Suture choice")
        if subj not in valid_subjects_objects or obj not in valid_subjects_objects:
            continue
        if rel not in valid_relations:
            continue
            
        key = (subj, rel, obj, nr['risk'].lower())
        new_explanation = nr.get('explanation', '').strip()
        
        if key not in rule_map:
            rule_map[key] = nr
        else:
            existing_exp = rule_map[key].get('explanation', '').strip()
            # Upgrade if current has no explanation but new one does
            if not existing_exp and new_explanation:
                rule_map[key]['explanation'] = new_explanation
            # Otherwise, keep original (per user request: one explanation is enough)
            
    return list(rule_map.values())

def main(args):
    print(f"Starting Production Knowledge Extraction. Model: {args.model}", flush=True)
    extractor = KnowledgeExtractor(model=args.model)
    
    all_extracted_rules = []
    
    if args.mock:
        all_extracted_rules = extractor.mock_extraction()
    else:
        if not os.path.exists(args.pdf_dir):
            print(f"PDF directory {args.pdf_dir} not found. Use --mock or create the directory.")
            return
            
        pdf_files = sorted([f for f in os.listdir(args.pdf_dir) if f.endswith('.pdf')])
        
        # Apply optional file limit
        if args.limit and args.limit > 0:
             print(f"Limiting processing to first {args.limit} files.")
             pdf_files = pdf_files[:args.limit]
             
        print(f"Found {len(pdf_files)} PDF files to process.")
             
        for i, pdf in enumerate(pdf_files):
            print(f"\n[{i+1}/{len(pdf_files)}] Processing {pdf}...", flush=True)
            text = extractor.extract_text_from_pdf(os.path.join(args.pdf_dir, pdf))
            if not text:
                 print(f"Skipping {pdf} (No text extracted)", flush=True)
                 continue
                 
            # Chunking text with 200 character overlap
            chunk_size = 4000
            overlap = 200
            chunks = []
            for j in range(0, len(text), chunk_size - overlap):
                chunks.append(text[j:j+chunk_size])
            
            print(f"  - Document split into {len(chunks)} chunks.", flush=True)
            
            for j, chunk in enumerate(chunks):
                print(f"  - Chunk {j+1}/{len(chunks)} | Rules so far: {len(all_extracted_rules)}", end='\r', flush=True)
                triplets = extractor.get_triplets_from_llm(chunk)
                if triplets:
                     all_extracted_rules = merge_rules(all_extracted_rules, triplets)
            print(f"\n  - Finished {pdf}. Unique rules in memory: {len(all_extracted_rules)}", flush=True)
                
    # Save extracted rules
    output_path = "src/rkg/extracted_rules.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_extracted_rules, f, indent=4)
    print(f"\nSuccess! Saved total {len(all_extracted_rules)} unique rules to {output_path}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_dir", type=str, default="/raid/manoranjan/rampreetham/medical_literature/")
    parser.add_argument("--model", type=str, default="llama3")
    parser.add_argument("--mock", action="store_true", help="Use seed knowledge instead of real LLM")
    parser.add_argument("--limit", type=int, default=0, help="Optional: Limit number of PDFs to process (0 = all)")
    args = parser.parse_args()
    main(args)


```

---

## File: `src/rkg/ontology.py`

```python
# Custom Ontology for Surgical Risk Prediction
# Schema: (Subject) -> [Relation/Condition] -> (Object) | [Risk Level] | [Intervention]

SURGICAL_ONTOLOGY = {
    "instruments": [
        "grasper", "bipolar", "hook", "scissors", "clipper", "irrigator", "null"
    ],
    "anatomy": [
        "liver", "gallbladder", "cystic_plate", "cystic_duct", "cystic_artery", 
        "cystic_pedicle", "blood_vessel", "fluid", "abdominal_wall_cavity", 
        "omentum", "gut", "specimen"
    ],
    "relations": [
        "grasp", "retract", "dissect", "coagulate", "clip", "cut", "aspirate", "wash", "null"
    ],
    "spatial_relations": [
        "above", "below", "left", "right", "within", "near", "touching"
    ],
    "risk_levels": ["None", "Low", "Medium", "High", "Critical"]
}

# Seed Knowledge (Rules of Safety)
# Format: (Subject, Relation, Object, Meta_Condition) -> Risk_Level, Explanation
SEED_KNOWLEDGE = [
    {
        "subject": "hook", 
        "relation": "coagulate", 
        "object": "cystic_duct", 
        "risk": "Critical",
        "explanation": "Cautery on the cystic duct can cause thermal injury and subsequent leakage or stricture."
    },
    {
        "subject": "hook", 
        "relation": "coagulate", 
        "object": "liver", 
        "risk": "Low",
        "explanation": "Normal hemostasis on the liver bed."
    },
    {
        "subject": "clipper", 
        "relation": "clip", 
        "object": "cystic_artery", 
        "risk": "None",
        "explanation": "Standard procedure: clipping the cystic artery before division."
    },
    {
        "subject": "grasper", 
        "relation": "retract", 
        "object": "gallbladder", 
        "risk": "None",
        "explanation": "Standard exposure: retracting gallbladder to visualize the Calot triangle."
    },
    {
        "subject": "hook", 
        "relation": "near", 
        "object": "cystic_artery", 
        "condition": "active_energy",
        "risk": "High",
        "explanation": "Thermal spread from the hook near the cystic artery may cause unintended bleeding if the vessel reaches critical temperature."
    },
    {
        "subject": "hook", 
        "relation": "near", 
        "object": "liver", 
        "condition": "active_energy",
        "risk": "Medium",
        "explanation": "Potential for unintended thermal injury to the liver parenchyma."
    }
]

```

---

## File: `src/rkg/graph_manager.py`

```python
import networkx as nx
import torch
import numpy as np
import os
import json
from .ontology import SURGICAL_ONTOLOGY, SEED_KNOWLEDGE

class RiskGraphManager:
    """
    Manages the Risk Knowledge Graph (RKG).
    Connects Instruments, Anatomy, and Actions to associated Risk Levels.
    """
    def __init__(self, use_sapbert=False, rules_path="src/rkg/extracted_rules.json"):
        self.graph = nx.MultiDiGraph()
        self.use_sapbert = use_sapbert
        self._build_initial_graph(rules_path)
        
        if self.use_sapbert:
            # We would load the SapBERT model here for semantic matching
            from transformers import AutoTokenizer, AutoModel
            try:
                 self.tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
                 self.model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext").cuda()
            except:
                 self.use_sapbert = False
                 print("Warning: SapBERT not found. Falling back to Symbolic matching.")

    def _build_initial_graph(self, rules_path):
        # Adding nodes from ontology
        for inst in SURGICAL_ONTOLOGY['instruments']:
            self.graph.add_node(inst, type='instrument')
        for anat in SURGICAL_ONTOLOGY['anatomy']:
            self.graph.add_node(anat, type='anatomy')
            
        # Adding Seed Edges (Hardcoded safely rules)
        for rule in SEED_KNOWLEDGE:
            self._add_rule_to_graph(rule)
            
        # Adding Extracted Edges (from LLM)
        if os.path.exists(rules_path):
            try:
                with open(rules_path, 'r') as f:
                    extracted_rules = json.load(f)
                    print(f"Loading {len(extracted_rules)} extracted rules from {rules_path}...")
                    for rule in extracted_rules:
                        self._add_rule_to_graph(rule)
            except Exception as e:
                print(f"Error loading extra rules: {e}")

    def _add_rule_to_graph(self, rule):
        # Ensure subject and object are in ontology/graph (simple filter)
        sub = rule['subject'].lower().strip()
        obj = rule['object'].lower().strip()
        rel = rule['relation'].lower().strip()
        
        # We only add if they match our base ontology nodes for safe matching
        # In a real SOTA, we would use semantic similarity to map to ontology
        if self.graph.has_node(sub) and self.graph.has_node(obj):
            self.graph.add_edge(
                sub, 
                obj, 
                relation=rel, 
                risk=rule['risk'], 
                explanation=rule['explanation'],
                condition=rule.get('condition', None)
            )
            
    def query_risk(self, subject, relation, target, condition=None):
        """
        Symbolic Matcher: Exact match for the triplet in the KG.
        """
        # Search for edge between subject and target with matching relation
        if not self.graph.has_node(subject) or not self.graph.has_node(target):
            return "None", "No knowledge about these entities."
            
        edge_data = self.graph.get_edge_data(subject, target)
        if edge_data:
            for key in edge_data:
                d = edge_data[key]
                if d['relation'] == relation:
                    # Check condition (e.g., active_energy)
                    if d['condition'] is not None:
                        if d['condition'] == condition:
                            return d['risk'], d['explanation']
                    else:
                        return d['risk'], d['explanation']
                        
        return "None", "No matching risk rule found."

    def semantic_query_risk(self, subject, relation, target, condition=None):
        """
        Embedding Matcher: Uses SapBERT (simulated) for synonym handling.
        """
        if not self.use_sapbert:
            return self.query_risk(subject, relation, target, condition)
            
        # 1. Encode query entities
        # 2. Compute similarity with KG nodes
        # 3. Retrieve most similar edge
        # [TODO: Implement full SapBERT embedding search]
        return self.query_risk(subject, relation, target, condition)

if __name__ == "__main__":
    rkg = RiskGraphManager()
    print("Testing Risk Query:")
    risk, explain = rkg.query_risk("hook", "coagulate", "cystic_duct")
    print(f"Result: {risk} | {explain}")
    
    risk2, explain2 = rkg.query_risk("hook", "near", "cystic_artery", condition="active_energy")
    print(f"Result (Near + Energy): {risk2} | {explain2} ")

```

---

## File: `src/reasoner_demo.py`

```python
import torch
import logging
import os
from dataset_ssg import get_dataloader_ssg
from models.tdt import TriDiffTransformer
from rkg.graph_manager import RiskGraphManager

def run_risk_reasoning_demo(model, dataloader, rkg, device, logger, num_frames=10):
    """
    Simulates a live surgical scene reasoning.
    Takes SSG outputs and queries the RKG for clinical significance.
    """
    model.eval()
    logger.info("--- Starting Intraoperative Risk Reasoning Demo ---")
    
    with torch.no_grad():
        for i, (frames, labels) in enumerate(dataloader):
            if i >= num_frames: break
            
            frames = frames.to(device)
            nodes = labels['nodes'].to(device)
            bboxes = labels['bboxes'].to(device)
            num_valid = labels['num_valid_nodes'][0].item()
            
            # 1. Perception Layer: Extract Scene Graph
            _, _, _, _, _, _, _, edge_logits, energy_logits = model(frames, nodes, bboxes)
            
            # 2. Extract Relationships (Edges) with high confidence
            edge_preds = torch.sigmoid(edge_logits[0]) # (N, N, C)
            energy_state = "active_energy" if torch.sigmoid(energy_logits[0]) > 0.5 else None
            
            logger.info(f"Frame {i}: Analyzing {num_valid} detected entities. Energy State: {energy_state}")
            
            # 3. Reasoning Layer: Match against RKG
            found_risks = []
            for subj_idx in range(num_valid):
                for obj_idx in range(num_valid):
                    if subj_idx == obj_idx: continue
                    
                    # Check each edge class
                    probs = edge_preds[subj_idx, obj_idx]
                    for rel_idx, prob in enumerate(probs):
                        if prob > 0.5: # Confidence threshold
                            subj_name = get_entity_name(nodes[0, subj_idx].item())
                            obj_name = get_entity_name(nodes[0, obj_idx].item())
                            rel_name = get_relation_name(rel_idx)
                            
                            # Query Knowledge Graph
                            risk, explanation = rkg.query_risk(subj_name, rel_name, obj_name, condition=energy_state)
                            
                            if risk != "None":
                                found_risks.append({
                                    "triplet": f"({subj_name}) - [{rel_name}] -> ({obj_name})",
                                    "risk": risk,
                                    "explanation": explanation
                                })
            
            # 4. Report Alerts
            if found_risks:
                logger.warning(f"⚠️ RISK DETECTED in Frame {i}!")
                for r in found_risks:
                    logger.warning(f"  - [{r['risk']}] {r['triplet']}")
                    logger.warning(f"    Reason: {r['explanation']}")
            else:
                logger.info(f"  - Scene is within safety margins.")
                
    logger.info("--- Reasoning Demo Complete ---")

def get_entity_name(idx):
    node_classes = [
        'grasper', 'bipolar', 'hook', 'scissors', 'clipper', 'irrigator', 
        'liver', 'gallbladder', 'cystic_plate', 'cystic_duct', 'cystic_artery', 
        'cystic_pedicle', 'blood_vessel', 'fluid', 'abdominal_wall_cavity', 
        'omentum', 'gut', 'specimen'
    ]
    if 0 <= idx < len(node_classes):
        return node_classes[idx]
    return "unknown"

def get_relation_name(idx):
    edge_classes = [
        'grasp', 'retract', 'dissect', 'coagulate', 'clip', 'cut', 'aspirate', 'wash', 'null',
        'above', 'below', 'left', 'right', 'horizontal', 'vertical', 'within', 'out_of', 'surround'
    ]
    if 0 <= idx < len(edge_classes):
        return edge_classes[idx]
    return "null"

if __name__ == "__main__":
    # Setup simple logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("Reasoner")
    
    # 1. Setup Environment
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 2. Load Knowledge Graph
    rkg = RiskGraphManager(rules_path="src/rkg/extracted_rules.json") 
    
    # 3. Load Model
    logger.info("Initializing Tri-Diff-Transformer (Plan A)...")
    model = TriDiffTransformer(plan='A').to(device)
    
    # Optional: Load pre-trained weights if they exist
    checkpoint_path = "logs/best_phase2_model.pt"
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading weights from {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint, strict=False)
            logger.info("Weights loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}. Proceeding with initialized weights.")
    
    # 4. Initialize Data (SSG-VQA)
    DATASET_DIR = "/raid/manoranjan/rampreetham/SSG-VQA"
    if os.path.exists(DATASET_DIR):
        logger.info(f"Initializing SSG-VQA Data from {DATASET_DIR}...")
        try:
            dataloader, _ = get_dataloader_ssg(DATASET_DIR, split='val', batch_size=1, num_workers=4)
            # 5. Run Demo
            run_risk_reasoning_demo(model, dataloader, rkg, device, logger, num_frames=10)
        except Exception as e:
            logger.error(f"Error during data loading or demo: {e}")
    else:
        logger.error(f"Dataset directory {DATASET_DIR} not found. Cannot run demo.")

```

---

