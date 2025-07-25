# PDF QA Generation Pipeline

## Overview
This pipeline processes PDF files to generate question-answer pairs using local LLMs. It supports both single PDF processing and batch processing of entire folders with resume capability.

## Features
- **Single PDF Processing**: Process individual PDF files
- **Batch Processing**: Process entire folders of PDFs
- **Resume Capability**: Restart from where you left off using checkpoints
- **Progress Tracking**: Automatic checkpoint system to track processed files

## Usage

### Single PDF Processing
```bash
python QA_generator/main.py path/to/single.pdf
```

### Batch Processing (Folder)
```bash
python QA_generator/main.py path/to/folder/containing/pdfs
```

### Resume Processing
```bash
# Resume from checkpoint (skip already processed files)
python QA_generator/main.py path/to/folder --resume

# Use custom checkpoint file
python QA_generator/main.py path/to/folder --checkpoint my_checkpoint.json
```

## How It Works

1. **PDF Parsing**: Uses Docling to convert PDFs to Markdown
2. **Question Generation**: Uses Qwen-2.5-7B to generate questions from chunks
3. **Answer Generation**: Uses Qwen-2.5-7B to generate answers for each question
4. **Checkpoint System**: Tracks progress to enable resuming interrupted batches

## Output Structure
```
qa_outputs/
├── file1_qa.json          # Final QA pairs
├── file2_qa.json
└── ...

question_outputs/
├── file1_questions.json   # Generated questions
├── file2_questions.json
└── ...

QA_generator/parser/parsed_outputs/
├── file1_parsed.md        # Parsed markdown
├── file2_parsed.md
└── ...
```

## Checkpoint File
The checkpoint file (`pipeline_checkpoint.json`) tracks:
- Which PDFs have been processed
- Output file locations
- Processing timestamps

## Examples

### Process all PDFs in the PDFs folder
```bash
python QA_generator/main.py PDFs/
```

### Process a single PDF
```bash
python QA_generator/main.py PDFs/The_iPhone_16.pdf
```

### Resume interrupted batch processing
```bash
python QA_generator/main.py PDFs/ --resume
