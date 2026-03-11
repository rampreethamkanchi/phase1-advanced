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
