# thesis-automotive-diagnostics

End-to-end Retrieval-Augmented Generation (RAG) system for automotive diagnostics. Integrates speech-to-text, query enrichment via synonym expansion, hybrid retrieval (BM25 + dense search), Reciprocal Rank Fusion (RRF), cross-encoder reranking, and LLM-based answer generation from vehicle manuals for accurate troubleshooting support.

## Requirements

- Python 3.10+
- CUDA-capable GPU (required for Mistral-7B and Whisper; tested on a T4 with 15GB VRAM)
- espeak (Linux/Colab) for text-to-speech output

## Installation

pip install -r requirements.txt  
apt-get install -y espeak  # Linux/Colab only  

## Setup (run once per vehicle manual)

python offline/icon_extraction.py --pdf path/to/manual.pdf --output data/icons/  
python offline/chunking.py --pdf path/to/manual.pdf --icons data/icons/ --output data/chunks/punto.json --vehicle punto  
python offline/indexing.py --chunks data/chunks/punto.json --chroma data/chroma_db/ --vehicle punto  

## Usage

python online/pipeline.py --vehicle punto --query "my car won't start"  
python online/pipeline.py --vehicle punto --audio recording.mp3 --speak  

## Evaluation

python evaluation/run_evaluation.py --ground_truth evaluation/ground_truth.json --output data/evaluation_results.json --latency   data/latency_records.json --vehicle punto  
python evaluation/metrics.py --ground_truth evaluation/ground_truth.json --results data/evaluation_results.json  

## Notebooks

The notebooks show the thesis results (retrieval metrics, latency analysis) and other steps that were used during development and testing.

## Data

- `data/chunks/punto.json` — pre-processed chunks for the Fiat Punto 2017 English manual (959 chunks, icon-injected)  
- `data/icons/` — icon sidecar files with manually annotated descriptions and PDF coordinates. Filenames reflect original extraction indices, not sequential numbering  
- `data/chroma_db/` is not included (generated locally via `indexing.py`)  
