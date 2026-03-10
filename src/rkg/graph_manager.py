import networkx as nx
import torch
import numpy as np
from .ontology import SURGICAL_ONTOLOGY, SEED_KNOWLEDGE

class RiskGraphManager:
    """
    Manages the Risk Knowledge Graph (RKG).
    Connects Instruments, Anatomy, and Actions to associated Risk Levels.
    """
    def __init__(self, use_sapbert=False):
        self.graph = nx.MultiDiGraph()
        self.use_sapbert = use_sapbert
        self._build_initial_graph()
        
        if self.use_sapbert:
            # We would load the SapBERT model here for semantic matching
            # For now, we simulate with a simple string-matching fallback
            from transformers import AutoTokenizer, AutoModel
            try:
                 self.tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
                 self.model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext").cuda()
            except:
                 self.use_sapbert = False
                 print("Warning: SapBERT not found. Falling back to Symbolic matching.")

    def _build_initial_graph(self):
        # Adding nodes from ontology
        for inst in SURGICAL_ONTOLOGY['instruments']:
            self.graph.add_node(inst, type='instrument')
        for anat in SURGICAL_ONTOLOGY['anatomy']:
            self.graph.add_node(anat, type='anatomy')
            
        # Adding edges (Rules)
        for rule in SEED_KNOWLEDGE:
            self.graph.add_edge(
                rule['subject'], 
                rule['object'], 
                relation=rule['relation'], 
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
