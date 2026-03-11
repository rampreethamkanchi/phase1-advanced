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
        """Calls local Ollama to extract triplets based on ontology."""
        prompt = f"""
        Extract surgical risk triplets from the following medical text.
        Use only these entity types: {SURGICAL_ONTOLOGY['instruments']} and {SURGICAL_ONTOLOGY['anatomy']}.
        Use only these relations: {SURGICAL_ONTOLOGY['relations']} or {SURGICAL_ONTOLOGY['spatial_relations']}.
        
        CRITICAL: Respond ONLY with a valid JSON list. No preamble or conversational filler.
        
        Format your output as a JSON list of objects:
        [
          {{"subject": "...", "relation": "...", "object": "...", "risk": "Low/Medium/High/Critical", "explanation": "..."}}
        ]
        
        Text:
        {text_chunk[:3000]} 
        """
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_ctx": 4096
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
                    return json.loads(json_str)
                return json.loads(clean_response)
        except Exception as e:
            print(f"LLM Error: {e}")
            return []
        return []

    def mock_extraction(self):
        """Returns seed knowledge as a mock result."""
        print("Using MOCK extraction mode...")
        return SEED_KNOWLEDGE

def main(args):
    extractor = KnowledgeExtractor(model=args.model)
    
    all_extracted_rules = []
    
    if args.mock:
        all_extracted_rules = extractor.mock_extraction()
    else:
        if not os.path.exists(args.pdf_dir):
            print(f"PDF directory {args.pdf_dir} not found. Use --mock or create the directory.")
            return
            
        pdf_files = [f for f in os.listdir(args.pdf_dir) if f.endswith('.pdf')]
        if args.limit and len(pdf_files) > args.limit:
             pdf_files = pdf_files[:args.limit]
             
        for i, pdf in enumerate(pdf_files):
            print(f"[{i+1}/{len(pdf_files)}] Processing {pdf}...", flush=True)
            text = extractor.extract_text_from_pdf(os.path.join(args.pdf_dir, pdf))
            if not text:
                 print(f"Skipping {pdf} (No text extracted)", flush=True)
                 continue
                 
            # Chunking text for LLM context window (simple split for demo)
            chunks = [text[j:j+3000] for j in range(0, len(text), 3000)]
            
            # For each PDF, we take first 5 chunks to avoid infinite loops during pilot
            chunk_limit = 5
            for j, chunk in enumerate(chunks[:chunk_limit]):
                print(f"  - Extracting from chunk {j+1}/{min(len(chunks), chunk_limit)}...", flush=True)
                triplets = extractor.get_triplets_from_llm(chunk)
                if triplets:
                     print(f"    + Found {len(triplets)} triplets.", flush=True)
                     all_extracted_rules.extend(triplets)
                
    # Save extracted rules
    output_path = "src/rkg/extracted_rules.json"
    with open(output_path, 'w') as f:
        json.dump(all_extracted_rules, f, indent=4)
    print(f"Saved total {len(all_extracted_rules)} rules to {output_path}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_dir", type=str, default="/raid/manoranjan/rampreetham/medical_literature/")
    parser.add_argument("--model", type=str, default="llama3")
    parser.add_argument("--mock", action="store_true", help="Use seed knowledge instead of real LLM")
    parser.add_argument("--limit", type=int, default=3, help="Limit number of PDFs to process")
    args = parser.parse_args()
    main(args)
