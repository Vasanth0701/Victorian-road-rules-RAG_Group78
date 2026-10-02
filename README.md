# WIL Project – Group 78

## Project
WIL Project – Victoria Road Rules Assistant RAG

## Group Information
- **Group ID:** 78
- **Course:** COSC2669/COSC2816

## Team Members

1. Vasanth Bas — s4136330
2. Siva Ishwarya Balamagesh — s4186954
3. Sri Varshan Rattapalivalasu Jayakumar — s4184004
4. Gayathri Devi Polavarapu — s4192325
5. Sai Sailesh Sugumaran — s4177969
6. Aleena Patrick — s4142243

## RAG Implementation

The project implements a Retrieval-Augmented Generation (RAG) pipeline for answering questions related to Victorian road rules.

The retrieval pipeline supports:
- BM25 sparse retrieval
- Dense embedding retrieval
- FAISS vector search
- Top-k document retrieval

## Local LLM Integration

A local Large Language Model is integrated with the retrieval pipeline using Ollama.

The RAG generation process is implemented in:

`src/rag_generate.py`

The system retrieves the top relevant road-rule passages and provides them as context to the local LLM. The model is instructed to generate answers using only the retrieved context.

## Running the RAG System

Ensure Ollama is installed and the required model is available locally.

Run the RAG system from the project root using:

```bash
python src/rag_generate.py
