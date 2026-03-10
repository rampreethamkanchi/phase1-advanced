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
        
        Format your output as a JSON list of objects:
        [
          {{"subject": "...", "relation": "...", "object": "...", "risk": "Low/Medium/High/Critical", "explanation": "..."}}
        ]
        
        Text:
        {text_chunk[:2000]} 
        """
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }
        
        try:
            response = requests.post(self.ollama_url, json=payload, timeout=60)
            if response.status_code == 200:
                result = response.json()
                return json.loads(result['response'])
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
        for pdf in pdf_files:
            print(f"Processing {pdf}...")
            text = extractor.extract_text_from_pdf(os.path.join(args.pdf_dir, pdf))
            # Chunking text for LLM context window (simple split for demo)
            chunks = [text[i:i+2000] for i in range(0, len(text), 2000)]
            for chunk in chunks[:3]: # Limit to first 3 chunks for test
                triplets = extractor.get_triplets_from_llm(chunk)
                all_extracted_rules.extend(triplets)
                
    # Save extracted rules
    output_path = "src/rkg/extracted_rules.json"
    with open(output_path, 'w') as f:
        json.dump(all_extracted_rules, f, indent=4)
    print(f"Saved {len(all_extracted_rules)} rules to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_dir", type=str, default="/raid/manoranjan/rampreetham/medical_literature/")
    parser.add_argument("--model", type=str, default="llama3")
    parser.add_argument("--mock", action="store_true", help="Use seed knowledge instead of real LLM")
    args = parser.parse_args()
    main(args)
