import argparse
import os
import json
import time
from pathlib import Path

from parser.parse import main as parse_main
from question_generator.generate import main as q_main
from answer_generator.generate import main as a_main

class PipelineManager:
    def __init__(self, checkpoint_file="pipeline_checkpoint.json"):
        self.checkpoint_file = checkpoint_file
        self.checkpoint = self.load_checkpoint()
        
    def load_checkpoint(self):
        """Load existing checkpoint if it exists"""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                print(f"Warning: Could not load checkpoint file, starting fresh")
        return {"processed_files": {}, "current_batch": {}}
    
    def save_checkpoint(self):
        """Save current checkpoint state"""
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(self.checkpoint, f, indent=2)
        except IOError as e:
            print(f"Warning: Could not save checkpoint: {e}")
    
    def is_file_processed(self, pdf_path):
        """Check if a PDF file has been fully processed"""
        pdf_name = Path(pdf_path).stem
        return pdf_name in self.checkpoint["processed_files"]
    
    def mark_file_processed(self, pdf_path, output_files):
        """Mark a PDF file as fully processed"""
        pdf_name = Path(pdf_path).stem
        self.checkpoint["processed_files"][pdf_name] = {
            "pdf_path": pdf_path,
            "completed_at": time.time(),
            "output_files": output_files
        }
        self.save_checkpoint()
    
    def get_unprocessed_pdfs(self, folder_path):
        """Get list of unprocessed PDF files in a folder"""
        pdf_files = []
        for file in os.listdir(folder_path):
            if file.lower().endswith('.pdf'):
                full_path = os.path.join(folder_path, file)
                if not self.is_file_processed(full_path):
                    pdf_files.append(full_path)
        return sorted(pdf_files)
    
    def process_single_pdf(self, pdf_path):
        """Process a single PDF through the entire pipeline with resume capability"""
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(pdf_path)}")
        print(f"{'='*60}")
        
        pdf_name = Path(pdf_path).stem
        
        try:
            # Initialize output paths
            md_path = os.path.join("QA_generator/parser/parsed_outputs", f"{pdf_name}_parsed.md")
            questions_output = os.path.join("question_outputs", f"{pdf_name}_questions.json")
            qa_output_file = os.path.join("qa_outputs", f"{pdf_name}_qa.json")
            
            # Step 1: Parse PDF to MD (only if markdown doesn't exist)
            if not os.path.exists(md_path):
                print("Step 1: Parsing PDF to Markdown...")
                parse_result = parse_main(pdf_path)
                if not parse_result:
                    print("Error: Failed to parse PDF")
                    return False
                print(f"✓ Parsed successfully: {md_path}")
            else:
                print(f"✓ Markdown already exists: {md_path}")
            
            # Step 2: Generate questions (only if questions file doesn't exist)
            if not os.path.exists(questions_output):
                print("\nStep 2: Generating questions...")
                os.makedirs("question_outputs", exist_ok=True)
                q_result = q_main(input_path=md_path, output=questions_output)
                if not q_result:
                    print("Error: Failed to generate questions")
                    return False
                print(f"✓ Questions generated: {questions_output}")
            else:
                print(f"✓ Questions already exist: {questions_output}")
            
            # Step 3: Generate answers (check if all answers are complete)
            all_questions_answered = False
            if os.path.exists(qa_output_file):
                # Check if the QA file has all the questions answered
                try:
                    with open(questions_output, 'r') as f:
                        questions_data = json.load(f)
                        total_questions = sum(len(item['questions']) for item in questions_data)
                    
                    with open(qa_output_file, 'r') as f:
                        answers_data = json.load(f)
                        answered_questions = len(answers_data)
                    
                    if answered_questions >= total_questions:
                        all_questions_answered = True
                        print(f"✓ Answers already complete: {qa_output_file} ({answered_questions}/{total_questions} questions)")
                    else:
                        print(f"⚠️ Partial answers found: {qa_output_file} ({answered_questions}/{total_questions} questions). Continuing generation...")
                except Exception as e:
                    print(f"⚠️ Could not verify answer completion status: {e}. Regenerating answers...")
            else:
                print("\nStep 3: Generating answers...")
                os.makedirs("qa_outputs", exist_ok=True)
            
            if not all_questions_answered:
                a_main(INPUT_FILE=questions_output, OUTPUT_FILE=qa_output_file)
                print(f"✓ Answers generated: {qa_output_file}")
            
            # Mark as processed (if not already marked)
            if pdf_name not in self.checkpoint["processed_files"]:
                output_files = {
                    "markdown": md_path,
                    "questions": questions_output,
                    "qa_pairs": qa_output_file
                }
                self.mark_file_processed(pdf_path, output_files)
            
            print(f"\n✅ Completed processing: {os.path.basename(pdf_path)}")
            return True
            
        except Exception as e:
            print(f"❌ Error processing {pdf_path}: {str(e)}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Process PDFs for QA generation")
    parser.add_argument("input_path", help="Path to PDF file or folder containing PDFs")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--checkpoint", default="pipeline_checkpoint.json", 
                       help="Checkpoint file path")
    args = parser.parse_args()
    
    # Initialize pipeline manager
    pipeline = PipelineManager(args.checkpoint)
    
    # Determine input type
    input_path = args.input_path
    
    if os.path.isfile(input_path) and input_path.lower().endswith('.pdf'):
        # Single PDF mode
        if pipeline.is_file_processed(input_path) and args.resume:
            print(f"✓ {os.path.basename(input_path)} already processed. Use --resume to reprocess.")
            return
        
        success = pipeline.process_single_pdf(input_path)
        if success:
            print("\n🎉 Pipeline completed successfully!")
        else:
            print("\n❌ Pipeline failed!")
            
    elif os.path.isdir(input_path):
        # Folder mode
        print(f"Scanning folder: {input_path}")
        unprocessed_pdfs = pipeline.get_unprocessed_pdfs(input_path)
        
        if not unprocessed_pdfs:
            print("✓ All PDFs in folder are already processed!")
            return
            
        print(f"Found {len(unprocessed_pdfs)} unprocessed PDFs")
        
        success_count = 0
        for pdf_path in unprocessed_pdfs:
            success = pipeline.process_single_pdf(pdf_path)
            if success:
                success_count += 1
            else:
                print(f"Skipping {os.path.basename(pdf_path)} due to error...")
                continue
        
        print(f"\n{'='*60}")
        print(f"Batch processing completed!")
        print(f"Successfully processed: {success_count}/{len(unprocessed_pdfs)} PDFs")
        print(f"{'='*60}")
        
    else:
        print("❌ Error: Input path must be a PDF file or folder containing PDFs")
        return

if __name__ == "__main__":
    main()
