import os

# Define the project root and the phases
PROJECT_ROOT = "."
OUTPUT_DIR = "research_docs"

# Mapping of Phases to their relevant files
# Note: Shared files like tdt.py are included in multiple phases where their code logic is relevant
PHASES = {
    "Phase_1_Triplet_Detection": [
        "src/dataset.py",
        "src/models/backbone.py",
        "src/models/query_decoder.py",
        "src/models/refiner.py",
        "src/losses/asl.py",
        "src/losses/mcl.py",
        "src/models/tdt.py",
        "src/train.py"
    ],
    "Phase_2_Phase_Detection": [
        "src/dataset_cholecT50.py",
        "src/models/t_encoder.py",
        "src/losses/supcon.py",
        "src/models/tdt.py",
        "src/train.py",
        "src/eval.py"
    ],
    "Phase_3_Scene_Graphs": [
        "src/dataset_ssg.py",
        "src/train_ssg.py",
        "src/losses/ssg_loss.py",
        "src/eval_ssg.py",
        "src/models/tdt.py"
    ],
    "Phase_4_Risk_Reasoning": [
        "src/rkg/extract_knowledge.py",
        "src/rkg/ontology.py",
        "src/rkg/graph_manager.py",
        "src/reasoner_demo.py"
    ]
}

def create_markdown_for_phase(phase_name, files):
    """Concatenates files into a single Markdown file."""
    output_path = os.path.join(OUTPUT_DIR, f"{phase_name}.md")
    
    with open(output_path, "w", encoding="utf-8") as out_f:
        out_f.write(f"# Methodology Context: {phase_name.replace('_', ' ')}\n\n")
        out_f.write("This document contains the complete source code for all modules contributing to this phase. "
                    "Use this as context for writing the methodology section of the research paper.\n\n")
        
        for file_path in files:
            full_path = os.path.join(PROJECT_ROOT, file_path)
            
            out_f.write(f"## File: `{file_path}`\n\n")
            
            if os.path.exists(full_path):
                try:
                    with open(full_path, "r", encoding="utf-8") as in_f:
                        content = in_f.read()
                        
                    # Determine language for markdown block
                    lang = "python" if file_path.endswith(".py") else ""
                    
                    out_f.write(f"```{lang}\n")
                    out_f.write(content)
                    out_f.write("\n```\n\n")
                    out_f.write("---\n\n") # Separator between files
                    
                except Exception as e:
                    out_f.write(f"**Error reading file:** {str(e)}\n\n")
            else:
                out_f.write(f"**Warning:** File `{file_path}` not found in project directory.\n\n")
                
    print(f"Successfully generated: {output_path}")

def main():
    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    print("Starting concatenation process...")
    for phase, files in PHASES.items():
        create_markdown_for_phase(phase, files)
    print("\nAll documents ready in the 'research_docs/' folder.")

if __name__ == "__main__":
    main()
