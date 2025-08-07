import time
import json
import requests
import os
from typing import List
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from QA_generator.answer_generator.managers.log_manager import LogManager

log_manager = LogManager()
logger = log_manager.get_logger()

# Add a lock for thread-safe operations
file_lock = threading.Lock()


def generate_answers(content: str, question: str, max_attempts: int = 3) -> str:
    """Generate answer with improved error handling and timeout management."""
    prompt = f"""Enhanced Assistant Prompt for Markdown Responses
    You are a knowledgeable assistant specializing in providing comprehensive, well-structured responses. Your primary objective is to analyze the provided content and deliver thorough, accurate answers to questions in proper Markdown format.
    Core Requirements
    Response Format

    All responses must be in Markdown format with proper heading hierarchy, formatting, and structure
    Use appropriate Markdown elements including:

    Headers (#, ##, ###) for organizing content
    Bold text for emphasis on key points
    Italics for subtle emphasis or technical terms
    Code blocks for technical references

    Blockquotes for important statements

    Lists (both ordered and unordered) for clarity
    Tables when presenting comparative data
    Links when referencing external concepts

    Content Standards

    Comprehensive Coverage: Provide detailed, in-depth responses that thoroughly address all aspects of the question
    Accuracy First: Ensure all information is precise and factually correct based on the provided content
    Descriptive Language: Use rich, descriptive language that paints a clear picture for the reader
    Logical Structure: Organize information in a logical flow that builds understanding progressively
    Examples and Applications: Include relevant examples, use cases, or practical applications where appropriate

    Writing Guidelines
    Language and Style

    Write exclusively in English unless the question specifically requires other languages
    Begin each sentence with a capital letter and use proper punctuation throughout
    Maintain a professional yet accessible tone
    Use active voice when possible for clarity and engagement

    Response Structure

    Direct Approach: Begin your response immediately with the relevant information
    Natural Flow: Write as if explaining the topic from your own expertise and understanding
    Comprehensive Detail: Aim for substantial responses that provide complete coverage of the topic
    Relevant Focus: Ensure all content directly relates to the question asked

    Prohibited Practices
    Avoid These Introductory Phrases:

    "According to the given context"
    "As per the text"
    "According to the content"
    "Based on the provided information"
    "The content states that"
    "From the given material"
    Any similar referential introductions

    Content Restrictions

    Do not include meta-commentary about the response process
    Avoid unnecessary explanations about your methodology
    Do not add context or information not relevant to the specific question
    Refrain from apologetic or uncertain language when the content provides clear answers

    Response Enhancement Techniques
    Structure Your Responses Using:

    Clear Headings to organize major topics
    Subheadings to break down complex concepts
    Bullet Points for listing features, benefits, or steps
    Numbered Lists for processes or ranked information
    Tables for comparing different elements
    Code Blocks for technical specifications or examples
    Blockquotes for highlighting critical information

    Content Enhancement Methods:

    Elaborate on key concepts with detailed explanations
    Provide context for technical terms or specialized knowledge
    Include practical examples to illustrate abstract concepts
    Connect related ideas to show broader implications
    Offer multiple perspectives when appropriate

    Input Processing
    Content: {content}
    Question: {question}
    """

    headers = {"Content-Type": "application/json"}

    for attempt in range(1, max_attempts + 1):
        try:
            # Reduce max_tokens to speed up generation
            max_tokens = 4000 if attempt == 1 else (3000 if attempt == 2 else 2000)
            
            payload = {
                "model": "base_model/Qwen2.5-7b-Instruct",
                "messages": [{"role": "system", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.5,  # Reduced temperature for faster, more focused responses
            }
            
            response = requests.post(
                "http://localhost:8096/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=120,  # Reduced timeout for faster failure detection
            )
            response.raise_for_status()
            
            data = response.json()
            if "choices" in data and data["choices"]:
                content = data["choices"][0]["message"]["content"].strip()
                if content and not content.startswith('Error'):
                    logger.info(f"✓ Generated answer on attempt {attempt}")
                    return content
                    
        except requests.exceptions.Timeout:
            logger.warning(f"[Attempt {attempt}] Request timeout - retrying")
            time.sleep(1)  # Shorter sleep
        except requests.exceptions.ConnectionError:
            logger.warning(f"[Attempt {attempt}] Connection error - server may be overloaded")
            time.sleep(2)  # Shorter sleep
        except requests.exceptions.HTTPError as e:
            logger.warning(f"[Attempt {attempt}] HTTP error {e.response.status_code}: {e}")
            time.sleep(1)
        except Exception as e:
            logger.warning(f"[Attempt {attempt}] Unexpected error: {e}")
            time.sleep(1)
    
    logger.error(f"Failed to generate answer after {max_attempts} attempts")
    return "Error: Could not generate answer after multiple attempts"


def process_single_question(args):
    """Process a single question - for use with threading."""
    content, question, question_id = args
    logger.info(f"Processing question {question_id}: {question[:60]}...")
    
    response = generate_answers(content, question)
    
    if response.startswith("Error:"):
        logger.warning(f"Failed to generate answer for question {question_id}")
        return {"prompt": question, "response": response, "failed": True}
    
    return {"prompt": question, "response": response, "failed": False}


def load_existing_answers(path: str) -> List[dict]:
    """Load existing answers with better error handling."""
    if not os.path.exists(path):
        logger.info(f"Output file {path} doesn't exist - starting fresh")
        return []
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            json_file = json.load(f)
            
        # Filter out error responses
        valid_answers = [
            pairs for pairs in json_file 
            if not str(pairs.get('response', 'Error')).strip().startswith('Error')
        ]
        
        logger.info(f"Loaded {len(valid_answers)} existing valid answers from {path}")
        return valid_answers
        
    except json.JSONDecodeError as e:
        logger.warning(f"Output file is corrupted: {e} - starting from scratch")
        # Backup the corrupted file
        backup_path = f"{path}.backup"
        if os.path.exists(path):
            os.rename(path, backup_path)
            logger.info(f"Corrupted file backed up to {backup_path}")
        return []
    except Exception as e:
        logger.error(f"Error loading existing answers: {e}")
        return []


def save_progress(output_data: List[dict], output_file: str) -> bool:
    """Save progress with error handling and thread safety."""
    try:
        with file_lock:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Write to temporary file first
            temp_file = f"{output_file}.tmp"
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            # Move temp file to final location (atomic operation)
            os.rename(temp_file, output_file)
            return True
        
    except Exception as e:
        logger.error(f"Error saving progress: {e}")
        return False


def main_sequential(INPUT_FILE: str, OUTPUT_FILE: str, SAVE_EVERY: int = 5):
    """Original sequential processing function - optimized version."""
    
    logger.info(f"Starting answer generation process (Sequential)")
    logger.info(f"Input: {INPUT_FILE}")
    logger.info(f"Output: {OUTPUT_FILE}")
    
    # Load input data
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded input data with {len(data)} sections")
    except Exception as e:
        logger.error(f"Failed to load input file: {e}")
        return

    # Load existing answers
    existing_answers = load_existing_answers(OUTPUT_FILE)
    answered_prompts = set(item["prompt"] for item in existing_answers)
    output_data = existing_answers.copy()

    # Prepare unanswered questions
    unanswered_questions = []
    total_questions = 0
    
    for item in data:
        content = item["content"]
        for question in item["questions"]:
            total_questions += 1
            if question not in answered_prompts:
                unanswered_questions.append((content, question))

    logger.info(f"Total questions: {total_questions}")
    logger.info(f"Already answered: {len(answered_prompts)}")
    logger.info(f"To process: {len(unanswered_questions)}")

    if not unanswered_questions:
        logger.info("All questions already answered!")
        return

    # Process questions sequentially with progress tracking
    counter = 0
    failed_count = 0
    
    for i, (content, question) in enumerate(tqdm(unanswered_questions, desc="Answering"), 1):
        logger.info(f"Processing question {i}/{len(unanswered_questions)}: {question[:60]}...")
        
        response = generate_answers(content, question)
        
        if response.startswith("Error:"):
            failed_count += 1
            logger.warning(f"Failed to generate answer for: {question[:60]}...")
        
        result = {"prompt": question, "response": response}
        output_data.append(result)
        answered_prompts.add(question)
        counter += 1

        # Save progress periodically
        if counter % SAVE_EVERY == 0:
            if save_progress(output_data, OUTPUT_FILE):
                logger.info(f"✓ Saved progress: {counter} answers processed ({failed_count} failed)")
            else:
                logger.error("Failed to save progress!")
        
        # Reduced pause to speed up processing
        time.sleep(0.5)

    # Final save
    if save_progress(output_data, OUTPUT_FILE):
        logger.info(f"✓ Final save complete!")
        logger.info(f"Total processed: {counter}")
        logger.info(f"Successful: {counter - failed_count}")
        logger.info(f"Failed: {failed_count}")
        logger.info(f"Success rate: {((counter - failed_count) / counter * 100):.1f}%")
    else:
        logger.error("Failed to save final results!")


def main_parallel(INPUT_FILE: str, OUTPUT_FILE: str, SAVE_EVERY: int = 5, max_workers: int = 3, batch_size: int = 4):
    """Parallel processing version with batch processing for better resource management."""
    
    logger.info(f"Starting answer generation process (Parallel with {max_workers} workers, batch size {batch_size})")
    logger.info(f"Input: {INPUT_FILE}")
    logger.info(f"Output: {OUTPUT_FILE}")
    
    # Load input data
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded input data with {len(data)} sections")
    except Exception as e:
        logger.error(f"Failed to load input file: {e}")
        return

    # Load existing answers
    existing_answers = load_existing_answers(OUTPUT_FILE)
    answered_prompts = set(item["prompt"] for item in existing_answers)
    output_data = existing_answers.copy()

    # Prepare unanswered questions with IDs
    unanswered_questions = []
    total_questions = 0
    
    for item in data:
        content = item["content"]
        for question in item["questions"]:
            total_questions += 1
            if question not in answered_prompts:
                unanswered_questions.append((content, question, len(unanswered_questions) + 1))

    logger.info(f"Total questions: {total_questions}")
    logger.info(f"Already answered: {len(answered_prompts)}")
    logger.info(f"To process: {len(unanswered_questions)}")

    if not unanswered_questions:
        logger.info("All questions already answered!")
        return

    # Process questions in batches
    counter = 0
    failed_count = 0
    total_processed = len(unanswered_questions)
    
    # Split questions into batches
    batches = [unanswered_questions[i:i + batch_size] for i in range(0, len(unanswered_questions), batch_size)]
    
    logger.info(f"Processing {len(batches)} batches of size {batch_size}")
    
    with tqdm(total=total_processed, desc="Answering") as pbar:
        for batch_idx, batch in enumerate(batches):
            logger.info(f"Processing batch {batch_idx + 1}/{len(batches)} (size: {len(batch)})")
            
            # Process current batch in parallel
            batch_results = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit tasks for current batch
                future_to_question = {
                    executor.submit(process_single_question, question_data): question_data 
                    for question_data in batch
                }
                
                # Collect results for current batch
                for future in as_completed(future_to_question):
                    try:
                        result = future.result()
                        batch_results.append(result)
                        
                        if result["failed"]:
                            failed_count += 1
                        
                    except Exception as e:
                        logger.error(f"Error processing question: {e}")
                        failed_count += 1

            # Process batch results
            for result in batch_results:
                # Remove the failed flag before saving
                result.pop("failed", None)
                output_data.append(result)
                answered_prompts.add(result["prompt"])
                counter += 1
                
                pbar.update(1)

            # Save progress periodically (after each batch or when SAVE_EVERY is reached)
            if (batch_idx + 1) % (SAVE_EVERY // batch_size or 1) == 0 or counter % SAVE_EVERY == 0:
                if save_progress(output_data, OUTPUT_FILE):
                    logger.info(f"✓ Saved progress: {counter} answers processed ({failed_count} failed)")
                else:
                    logger.error("Failed to save progress!")

    # Final save
    if save_progress(output_data, OUTPUT_FILE):
        logger.info(f"✓ Final save complete!")
        logger.info(f"Total processed: {counter}")
        logger.info(f"Successful: {counter - failed_count}")
        logger.info(f"Failed: {failed_count}")
        logger.info(f"Success rate: {((counter - failed_count) / counter * 100):.1f}%")
    else:
        logger.error("Failed to save final results!")


# Convenience function to choose processing method
def main(INPUT_FILE: str, OUTPUT_FILE: str, SAVE_EVERY: int = 5, use_parallel: bool = False, max_workers: int = 3, batch_size: int = 4):
    """Main function that chooses between sequential and parallel processing."""
    if use_parallel:
        main_parallel(INPUT_FILE, OUTPUT_FILE, SAVE_EVERY, max_workers, batch_size)
    else:
        main_sequential(INPUT_FILE, OUTPUT_FILE, SAVE_EVERY)


if __name__ == "__main__":
    INPUT_FILE = "question_outputs/Clockmaker_questions.json"
    OUTPUT_FILE = "qa_outputs/Clockmaker_qa.json"
    SAVE_EVERY = 5
    
    # Choose processing method
    USE_PARALLEL = True  # Set to False for sequential processing
    MAX_WORKERS = 2  # Reduce if your server gets overloaded
    BATCH_SIZE = 4  # Batch size for parallel processing
    
    main(INPUT_FILE, OUTPUT_FILE, SAVE_EVERY, USE_PARALLEL, MAX_WORKERS, BATCH_SIZE)
