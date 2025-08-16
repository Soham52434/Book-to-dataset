Here’s the polished README.md content, fully formatted in Markdown so you can copy–paste directly:

# Book → Dataset Pipeline (Local, No S3)

This repository provides a **lightweight, S3-free, and GPU/CPU-friendly pipeline** for converting books (in PDF format) into clean, structured **datasets**. The processed datasets can be optionally indexed for **retrieval-augmented generation (RAG)** workflows using FAISS, BM25/TF-IDF, or a remote vector database such as Pinecone.

---

## Processing Stages

1. **Parse PDF** → extract raw page text  
   *[`modules/parse_pdf.py`]*  
2. **Normalize and clean** → remove boilerplate, join hyphenated words  
   *[`modules/normalize_content.py`]*  
3. **Structure detection** → identify section headings  
   *[`modules/structure_detect.py`]*  
4. **Chunking** → segment text into overlapping blocks  
   *[`modules/chunking.py`]*  
5. **Embeddings (optional)** → generate dense vector embeddings  
   *[`modules/embeddings.py`]*  
6. **Indexing (optional)** → build FAISS and/or BM25/TF-IDF indices  
   *[`modules/bm25_index.py`]*  
7. **Quality Control** → compute coverage and basic statistics  
   *[`modules/qc_checks.py`]*  
8. **Export** → save datasets in JSONL and Parquet formats  

> **Note:** This pipeline does not rely on S3 and has no mandatory cloud dependencies.  

---

## Quick Start

```bash
# 0. Use Python 3.9 or later
pip install -r requirements.txt

# 1. Place your PDF in the working directory
# Example: data/work/book.pdf
# (or update the path in the configuration file)

# 2. Run the pipeline
python app.py --config data/config/example.json

Outputs:
	•	Dataset: data/work/chunks.jsonl (+ .parquet if pyarrow is installed)
	•	Indices: data/indices/ (BM25/TF-IDF and/or FAISS if enabled)
	•	Report: data/reports/report.json

⸻

Configuration

An example configuration is provided in data/config/example.json.

{
  "paths": {
    "input_pdf": "data/work/book.pdf",
    "work_dir": "data/work",
    "indices_dir": "data/indices",
    "reports_dir": "data/reports"
  },
  "parse": {
    "ocr_if_needed": false
  },
  "chunking": {
    "target_chars": 1200,
    "overlap": 120
  },
  "embeddings": {
    "enabled": true,
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "device": "cpu"
  },
  "bm25": { "enabled": true },
  "faiss": { "enabled": false, "metric": "ip" }
}

Key Options
	•	OCR: Disabled by default. If the PDF consists of scanned images, set "ocr_if_needed": true and install pytesseract along with the Tesseract binary.
	•	FAISS: Disabled by default. Enable if faiss-cpu is installed and desired.

⸻

Notes
	•	The pipeline is designed for local execution (desktop, Kaggle, or Colab) without S3.
	•	If an embedding model cannot be downloaded (e.g., due to restricted internet access), set "embeddings.enabled": false and rely on BM25/TF-IDF indices.
	•	For academic or highly structured PDFs, tools such as GROBID or Unstructured may provide richer parsing.

⸻

Running with LLaMA 3 (via vLLM) on Colab / A100

pip install -r requirements.txt
pip install vllm

# Start vLLM (new cell / background process)
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --dtype float16 --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --host 0.0.0.0 --port 8000 &

# Environment variables for pipeline
export LLM_BASE_URL=http://127.0.0.1:8000/v1
export LLM_API_KEY=local
export LLM_MODEL=meta-llama/Meta-Llama-3-8B-Instruct

# Enable LLM in configuration
# "llm": {"enabled": true, "sectionize": true, "chunk_hints": false, "qa_pairs": true}

# Run pipeline
python app.py --config data/config/example.json

AWS (Single Node, A100)

Run the same vLLM server on your AWS instance. For multi-GPU setups, specify --tensor-parallel-size > 1.

⸻

Pinecone Integration (Optional, Remote Vector Database)
	1.	Install the client library

pip install pinecone-client


	2.	Set credentials

export PINECONE_API_KEY=YOUR_KEY


	3.	Update configuration (data/config/example.json)

"vectordb": {
  "provider": "pinecone",
  "api_key": "${PINECONE_API_KEY}",
  "index_name": "book-index-demo",
  "environment": "us-east-1",
  "metric": "cosine",
  "namespace": "default"
}


	4.	Run the pipeline
After embeddings are computed, chunks will be automatically upserted to Pinecone with metadata (section, page range, and text).
