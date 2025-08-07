import os
import json
import time
import re
import requests
from typing import List, Optional
from loguru import logger

from question_generator.managers.log_manager import LogManager
from question_generator.managers.vllm_manager import (
    check_vLLM as check_vllm_manager,
    start_vllm_server as start_vllm_server_manager,
)  

class QuestionGenerator:
    """Main question generator class using vLLM."""

    def __init__(
        self,
        model_path: str = "base_model/Qwen2.5-7b-Instruct",
        host: str = "localhost",
        port: int = 8096,
        model_url: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        self.model_path = model_path
        self.host = host
        self.port = port
        self.model_url = model_url or f"http://{self.host}:{self.port}"
        self.model_name = model_name or model_path
        self.logger = LogManager().get_logger(__name__)
        self.global_questions = set()

    def check_vLLM(self, port: Optional[int] = None) -> bool:
        """Check if vLLM server is running."""
        return check_vllm_manager(port or self.port)

    def start_vllm_server(self) -> bool:
        """Start the vLLM server if not already running."""
        return start_vllm_server_manager(
            model_path=self.model_path,
            host=self.host,
            port=self.port
        )

    def extract_json_from_response(self, response_text: str) -> Optional[List[dict]]:
        response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
        response_text = re.sub(r'({"question":"[^"]*"),\s*"question":"', r'\1},{"question":"', response_text)
        if not response_text.startswith('['):
            response_text = f'[{response_text}'
        if not response_text.endswith(']'):
            response_text = f'{response_text}]'
        try:
            questions = json.loads(response_text)
            if isinstance(questions, list):
                return questions
            else:
                logger.error("Response is not a JSON array")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON: {str(e)} | Snippet: {response_text[:100]}")
        return None

    def deduplicate_questions(self, questions: List[dict]) -> List[dict]:
        unique = []
        local_seen = set()
        for q in questions:
            q_text = q.get("question", "")
            if not q_text or q_text in local_seen or q_text in self.global_questions:
                continue
            unique.append(q)
            local_seen.add(q_text)
            self.global_questions.add(q_text)
        return unique

    def generate_questions(self, content: str, section_number: int, max_attempts: int = 3) -> List[dict]:
        """Use only the /v1/chat/completions endpoint to generate questions."""
        base_prompt = f"""
        You are a specialized question generation expert. Your task is to analyze section {section_number} and create comprehensive questions that thoroughly cover all content aspects.

## Core Requirements:
- STRICTLY Generate around 50-60 unique questions or more if possible
- Each question must address different content elements
- Ensure zero overlap with questions from other sections
- Use varied question structures and cognitive levels

## Question Diversity Guidelines:
- **Factual recall**: "What is...", "Define...", "List..."
- **Conceptual understanding**: "Explain how...", "Why does..."
- **Application**: "How would you apply...", "In what scenarios..."
- **Analysis**: "Compare...", "What are the implications of..."
- **Synthesis**: "How do X and Y relate...", "What connections exist..."
- **Evaluation**: "What are the advantages/disadvantages...", "Assess..."

## JSON Output Requirements:
- Return ONLY valid JSON array
- No markdown code blocks, explanations, or additional text
- Each object must contain exactly one "question" key
- Questions must end with appropriate punctuation
- Use double quotes for all strings
- Ensure proper JSON escaping for quotes within questions

## Format (STRICT):
[{{"question": "First question here?"}}, {{"question": "Second question here?"}}, {{"question": "Third question here?"}}]

## Content Analysis Instructions:
1. Identify all key concepts, terms, and ideas
2. Note relationships between different elements
3. Consider practical applications and real-world examples
4. Include both surface-level and deeper analytical questions
5. Address cause-and-effect relationships where applicable

Content: {content}
        """.strip()

        headers = {"Content-Type": "application/json"}

        for attempt in range(1, max_attempts + 1):
            try:
                payload = {
                    "model": self.model_name,
                    "messages": [{"role": "system", "content": base_prompt}],
                    "max_tokens": 6000 if attempt == 1 else 4000,
                    "temperature": 0.7,
                }
                response = requests.post(
                    f"{self.model_url}/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=600
                )
                response.raise_for_status()
                res = response.json()
                content = res["choices"][0]["message"]["content"]
                questions = self.extract_json_from_response(content)
                if questions:
                    deduped = self.deduplicate_questions(questions)
                    logger.success(f"Generated {len(deduped)} questions for section {section_number} using chat/completions")
                    return deduped
            except Exception as e:
                logger.error(f"Chat completions attempt {attempt}: {str(e)}")
                time.sleep(3)

        logger.error(f"Chat completions failed for section {section_number}")
        return []

    def split_md_into_chunks(self, content: str, output_dir: str = "md_chunks") -> List[str]:
        sections = []
        current_section = []

        os.makedirs(output_dir, exist_ok=True)

        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('#'):
                if current_section and any(l.strip() for l in current_section[1:]):
                    chunk = '\n'.join(current_section)
                    sections.append(chunk)
                    # Save chunk to file
                    with open(os.path.join(output_dir, f"chunk_{len(sections)}.md"), 'w', encoding='utf-8') as f:
                        f.write(chunk)
                current_section = [line]
            else:
                current_section.append(line)

        if current_section and any(l.strip() for l in current_section[1:]):
            chunk = '\n'.join(current_section)
            sections.append(chunk)
            with open(os.path.join(output_dir, f"chunk_{len(sections)}.md"), 'w', encoding='utf-8') as f:
                f.write(chunk)

        logger.info(f"Split content into {len(sections)} sections and saved to '{output_dir}'")
        return sections


    def process_md(self, file_path: str, output_file: str):
        logger.info(f"Processing {file_path}")
        if not self.check_vLLM():
            logger.error("vLLM not available.")
            return

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        sections = self.split_md_into_chunks(content, output_dir="md_chunks")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        all_data = []

        for i, section in enumerate(sections, 1):
            questions = self.generate_questions(section, i)
            if questions:
                entry = {
                    "content": section,
                    "questions": [q["question"] for q in questions],
                    "source_file": os.path.basename(file_path)
                }
                all_data.append(entry)
                logger.success(f"Added questions for section {i}")
            time.sleep(1)

        pdf_name = os.path.splitext(os.path.basename(file_path))[0]
        if pdf_name.endswith("_parsed"):
            pdf_name = pdf_name[:-7]  # remove '_parsed' suffix if present

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)

        logger.success(f"Saved all questions to {output_file}")

    def process_directory(self, dir_path: str, output_dir: str):
        files = [f for f in os.listdir(dir_path) if f.endswith(".md")]
        logger.info(f"Found {len(files)} markdown files in {dir_path}")
        for file in files:
            self.process_md(os.path.join(dir_path, file), output_dir)


def main(input_path: str, output: str):
    print(f"Running question generator on: {input_path}")
    print(f"Saving questions to: {output}")

    generator = QuestionGenerator(
        model_path="base_model/Qwen2.5-7b-Instruct",
        host="localhost",
        port=8096
    )

    if not generator.check_vLLM():
        generator.logger.info("vLLM not running. Starting it now...")
        if not generator.start_vllm_server():
            generator.logger.error("Failed to start vLLM.")
            exit(1)

    generator.process_md(input_path, output)

    return {
        "output_file": output
    }


if __name__ == "__main__":
    generator = QuestionGenerator(
        model_path="base_model/Qwen2.5-7b-Instruct",
        host="localhost",
        port=8096
    )

    if not generator.check_vLLM():
        generator.logger.info("vLLM not running. Starting it now...")
        if not generator.start_vllm_server():
            generator.logger.error("Failed to start vLLM.")
            exit(1)

    # Save output JSON as question_outputs/{pdf_name}.json
    generator.process_md("TravelEnglish1_parsed.md", "question_outputs")
