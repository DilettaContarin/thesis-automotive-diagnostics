## thesis-automotive-diagnostics
End-to-end Retrieval-Augmented Generation (RAG) system for automotive diagnostics. Integrates speech-to-text, query enrichment via synonym expansion, hybrid retrieval (BM25 + dense search), Reciprocal Rank Fusion (RRF), cross-encoder reranking, and LLM-based answer generation from vehicle manuals for accurate troubleshooting support.

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
python evaluation/run_evaluation.py --ground_truth evaluation/ground_truth.json --output data/evaluation_results.json --latency data/latency_records.json --vehicle punto
python evaluation/metrics.py --ground_truth evaluation/ground_truth.json --results data/evaluation_results.json
