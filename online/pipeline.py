"""
pipeline.py
-----------
Main online pipeline orchestrator — vehicle session management and unified
end-to-end pipeline function.

Session state:
  The pipeline maintains a single active vehicle per session. Switching
  vehicles reloads all indexes (chunks.json, BM25, ChromaDB collection).
  Global variables are used deliberately for the Colab/single-session context;
  a production deployment would encapsulate this in a Pipeline class.

Pipeline flow:
  Audio (optional) → ASR → cleaning → enrichment → retrieval → reranking
  → generation → TTS (optional)

Three experimental conditions (for evaluation):
  1. Baseline:   raw query → retrieve_all → top-5 RRF → generate
  2. Enriched:   enriched query → retrieve_all → top-5 RRF → generate
  3. Reranked:   enriched query → retrieve_all → top-10 RRF → rerank → top-5 → generate

Usage:
    from pipeline import pipeline, load_vehicle

    load_vehicle("punto")
    pipeline(query="my car won't start")
    pipeline(audio_path="recording.mp3", speak_answer=True)
"""

import json
import os
import time

import numpy as np
import chromadb
from rank_bm25 import BM25Okapi

import retrieval as ret
from query_analysis import clean_query, enrich_query
from generation import generate
from tts import speak_and_play
from online.asr import transcribe as asr_transcribe


# ── Paths (edit or override via environment variables) ───────────────────────
CHUNKS_DIR  = os.environ.get("CHUNKS_DIR",  "data/chunks")
CHROMA_PATH = os.environ.get("CHROMA_PATH", "data/chroma_db")
# ─────────────────────────────────────────────────────────────────────────────

# ── Session state ─────────────────────────────────────────────────────────────
current_vehicle = None
_chroma_client  = None


def _get_client():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _chroma_client


# ── Vehicle management ────────────────────────────────────────────────────────

def load_vehicle(vehicle_name):
    """
    Load all indexes for the specified vehicle.

    Sets module-level state in retrieval.py:
      - retrieval.chunks     ← loaded from {CHUNKS_DIR}/{vehicle_name}.json
      - retrieval.bm25       ← rebuilt from chunks (takes ~2 seconds)
      - retrieval.collection ← ChromaDB collection for this vehicle

    BM25 is rebuilt each session (cannot be persisted). ChromaDB loads
    from disk instantly since it was built offline by indexing.py.

    Args:
        vehicle_name (str): must match both a chunks JSON file and a
                            ChromaDB collection name (e.g. "punto_2017")
    """
    global current_vehicle

    chunks_path = os.path.join(CHUNKS_DIR, f"{vehicle_name}.json")
    if not os.path.exists(chunks_path):
        raise FileNotFoundError(
            f"Chunks file not found: {chunks_path}\n"
            f"Run offline/chunking.py --vehicle {vehicle_name} first."
        )

    # Load chunks
    with open(chunks_path, encoding="utf-8") as f:
        ret.chunks = json.load(f)
    print(f"Loaded {len(ret.chunks)} chunks from {chunks_path}")

    # Rebuild BM25 index
    tokenized = [c["text"].lower().split() for c in ret.chunks]
    ret.bm25  = BM25Okapi(tokenized)
    print(f"BM25 index ready — {len(tokenized)} documents")

    # Load ChromaDB collection
    client         = _get_client()
    existing       = [c.name for c in client.list_collections()]
    if vehicle_name not in existing:
        raise ValueError(
            f"ChromaDB collection '{vehicle_name}' not found.\n"
            f"Run offline/indexing.py --vehicle {vehicle_name} first."
        )
    ret.collection = client.get_collection(vehicle_name)
    print(f"ChromaDB collection '{vehicle_name}' loaded — {ret.collection.count()} chunks")

    current_vehicle = vehicle_name
    print(f"Vehicle set to: {current_vehicle}")


def check_vehicle():
    """
    Ensure a vehicle is loaded. Prompts for input if not.
    Returns True if vehicle is set, False otherwise.
    """
    global current_vehicle
    if current_vehicle is not None:
        return True
    vehicle = input("Please specify your vehicle (e.g. 'punto_2017'): ").strip()
    if vehicle:
        load_vehicle(vehicle)
        return True
    print("No vehicle specified — cannot proceed.")
    return False


# ── Timing utility ────────────────────────────────────────────────────────────

class _Timer:
    """Context manager for timing a named stage."""
    def __init__(self, label, timings_dict):
        self.label   = label
        self.timings = timings_dict

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *args):
        self.timings[self.label] = time.time() - self._start


def _timer(label, timings_dict):
    return _Timer(label, timings_dict)


def print_timings(timings):
    total = sum(timings.values())
    print(f"\n{'='*65}")
    print("  LATENCY BREAKDOWN")
    print(f"{'='*65}")
    for stage, t in timings.items():
        bar = "█" * int((t / total) * 30) if total > 0 else ""
        print(f"  {stage:<30} {t:.3f}s  {bar}")
    print(f"  {'TOTAL':<30} {total:.3f}s")


# ── Display utilities ─────────────────────────────────────────────────────────

def print_results(label, results_dict, show_scores=True):
    print(f"\n")
    print(f"  {label}")
    print(f"\n")
    for stage in ["bm25", "dense", "rrf", "reranked"]:
        if stage not in results_dict:
            continue
        print(f"\n--- {stage.upper()} ---")
        for i, r in enumerate(results_dict[stage]):
            page      = r["metadata"].get("page_number", "?")
            score     = r.get("rerank_score", r.get("score", 0))
            score_str = f"| score {score:.3f} " if show_scores else ""
            print(f"  [{i+1}] p{page} {score_str}| {r['text'][:100]}...")


def print_answer(label, answer):
    print(f"\n")
    print(f"  ANSWER — {label}")
    print(f"\n")
    print(answer)


# ── Main pipeline function ────────────────────────────────────────────────────

def pipeline(
    query            = None,
    audio_path       = None,
    use_enrichment   = True,
    generate_answers = True,
    speak_answer     = False,
    top_k            = 5,
    measure_latency  = True,
):
    """
    Full end-to-end diagnostic pipeline.

    Input modes:
        query="typed question"       — text input
        audio_path="/path/to/file"   — audio input via Whisper ASR

    Flags:
        use_enrichment=True    — apply SYNONYM_MAP vocabulary enrichment
                                 (Enhancement 1; ablation condition 2)
        use_enrichment=False   — baseline: raw query only
                                 (ablation condition 1)
        generate_answers=True  — run Mistral generation (slow, ~15-20s)
        generate_answers=False — retrieval only (fast, for quick testing)
        speak_answer=True      — synthesise final answer via espeak
        measure_latency=True   — print per-stage timing breakdown

    Returns:
        timings (dict): stage latencies in seconds, or empty dict if
                        measure_latency=False
    """
    if not check_vehicle():
        return {}

    timings = {}

    # ── Stage 1: ASR ──────────────────────────────────────────────────────────
    if audio_path is not None:
        print(f"Transcribing: {audio_path}")
        with _timer("asr", timings):
            raw_query = asr_transcribe(audio_path)
        print(f"Transcript: {raw_query}")
    elif query is not None:
        raw_query = query
    else:
        print("ERROR: provide either query= or audio_path=")
        return {}

    # ── Stage 2: Query analysis ───────────────────────────────────────────────
    with _timer("query_analysis", timings):
        enriched, matched = enrich_query(raw_query)

    print(f"\nORIGINAL:  {raw_query}")
    if use_enrichment and matched:
        print(f"ENRICHED:  {enriched}")
        print(f"TRIGGERED: {matched}")
    elif use_enrichment and not matched:
        print("ENRICHED:  (no enrichment applied — query already technical)")
    else:
        print("ENRICHED:  (enrichment disabled)")

    # ── Stage 3: Retrieval — raw query (baseline condition) ───────────────────
    with _timer("retrieval_raw", timings):
        raw_results             = ret.retrieve_all(
            raw_query, top_k=top_k, timings=timings, prefix="raw_"
        )
        raw_reranked            = ret.rerank(raw_query, list(raw_results["rrf"]))[:top_k]
        raw_results["reranked"] = raw_reranked
    print_results("RETRIEVAL — RAW QUERY", raw_results)

    # ── Stage 4: Retrieval — enriched query (Enhancement 1) ──────────────────
    if use_enrichment and matched:
        with _timer("retrieval_enriched", timings):
            enriched_results             = ret.retrieve_all(
                enriched, top_k=top_k, timings=timings, prefix="enriched_"
            )
            enriched_reranked            = ret.rerank(enriched, list(enriched_results["rrf"]))[:top_k]
            enriched_results["reranked"] = enriched_reranked
        print_results("RETRIEVAL — ENRICHED QUERY", enriched_results)
    else:
        enriched_results = raw_results  # no enrichment fired, reuse raw

    # ── Stage 5: Generation ───────────────────────────────────────────────────
    best_answer = None

    if generate_answers:
        # Condition 1: RRF only, raw query (baseline)
        with _timer("generation_rrf_raw", timings):
            ans_rrf_raw = generate(raw_query, raw_results["rrf"])
        print_answer("RRF | raw query", ans_rrf_raw)

        # Condition 2: Reranked, raw query
        with _timer("generation_reranked_raw", timings):
            ans_reranked_raw = generate(raw_query, raw_results["reranked"])
        print_answer("Reranked | raw query", ans_reranked_raw)

        if use_enrichment and matched:
            # Condition 3: RRF only, enriched query (Enhancement 1)
            with _timer("generation_rrf_enriched", timings):
                ans_rrf_enriched = generate(enriched, enriched_results["rrf"])
            print_answer("RRF | enriched query", ans_rrf_enriched)

            # Condition 4: Reranked + enriched (Enhancement 1 + 2 combined)
            with _timer("generation_reranked_enriched", timings):
                ans_reranked_enriched = generate(enriched, enriched_results["reranked"])
            print_answer("Reranked | enriched query", ans_reranked_enriched)

            best_answer = ans_reranked_enriched
        else:
            best_answer = ans_reranked_raw

    else:
        print("\n(generation skipped — set generate_answers=True to run Mistral)")

    # ── Stage 6: TTS ─────────────────────────────────────────────────────────
    if speak_answer and best_answer:
        print("\n Speaking answer...")
        speak_and_play(best_answer)

    # ── Latency summary ───────────────────────────────────────────────────────
    if measure_latency and timings:
        print_timings(timings)

    return timings


# ── CLI entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the automotive diagnostics pipeline")
    parser.add_argument("--vehicle",  required=True, help="Vehicle name (e.g. punto_2017)")
    parser.add_argument("--query",    default=None,  help="Text query")
    parser.add_argument("--audio",    default=None,  help="Path to audio file")
    parser.add_argument("--no-enrich",    action="store_true", help="Disable enrichment (baseline)")
    parser.add_argument("--no-generate",  action="store_true", help="Skip generation (retrieval only)")
    parser.add_argument("--speak",        action="store_true", help="Enable TTS output")
    args = parser.parse_args()

    load_vehicle(args.vehicle)
    pipeline(
        query            = args.query,
        audio_path       = args.audio,
        use_enrichment   = not args.no_enrich,
        generate_answers = not args.no_generate,
        speak_answer     = args.speak,
    )
