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
