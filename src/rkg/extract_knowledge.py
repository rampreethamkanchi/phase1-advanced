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

