"""
run_evaluation.py
-----------------
Batch evaluation runner for the automotive diagnostics RAG pipeline.

Runs all 50 ground truth queries through three retrieval conditions:
  - raw:      baseline hybrid retrieval (BM25 + dense + RRF, no enrichment)
  - enriched: retrieval with vocabulary enrichment (Enhancement 1)
  - reranked: enriched query + cross-encoder reranking (Enhancement 2)

Saves results incrementally after each query (crash-safe on Colab).
Latency is recorded per stage per query.

Output files:
  evaluation_results.json  — answers and retrieved pages per condition per query
  latency_records.json     — per-stage timings per query

Usage:
    python evaluation/run_evaluation.py \
        --ground_truth  evaluation/ground_truth.json \
        --output        data/evaluation_results.json \
        --latency       data/latency_records.json \
        --vehicle       punto
"""

import json
import os
import time
import argparse
import numpy as np


# ── Timing utility ───────────────────────────────────────────────────────────

class _Timer:
    def __init__(self, label, timings_dict):
        self.label   = label
        self.timings = timings_dict

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *args):
        self.timings[self.label] = time.time() - self._start


def timer(label, timings_dict):
    return _Timer(label, timings_dict)


# ── Batch evaluation ──────────────────────────────────────────────────────────

def run_evaluation(
    ground_truth_path,
    output_path,
    latency_path,
    top_k = 5,
):
    """
    Run the pipeline on all ground truth queries, collect results and latency.

    Conditions evaluated per query:
      raw      → clean query → BM25 + dense → RRF top-5 → generate
      enriched → enriched query → BM25 + dense → RRF top-5 → generate
      reranked → enriched query → BM25 + dense → RRF top-10 → rerank → top-5 → generate

    Args:
        ground_truth_path: path to ground_truth.json
        output_path:       where to save evaluation_results.json
        latency_path:      where to save latency_records.json
        top_k:             number of chunks passed to generation

    Saves incrementally after every query so Colab crashes don't lose progress.
    """
    # Import here to avoid circular dependency when used as a module
    from online/pipeline import load_vehicle, check_vehicle
    from online/retrieval import retrieve_all, rerank
    from online/generation import generate
    from online/query_analysis import enrich_query

    if not check_vehicle():
        return

    with open(ground_truth_path, encoding="utf-8") as f:
        ground_truth = json.load(f)

    # Resume if output already exists
    if os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as f:
            results = json.load(f)
        print(f"Resuming — {len(results)} queries already done")
    else:
        results = {}

    if os.path.exists(latency_path):
        with open(latency_path, encoding="utf-8") as f:
            latency_records = json.load(f)
    else:
        latency_records = []

    already_done = {r["query_key"] for r in latency_records}
    total        = len(ground_truth)

    for idx, entry in enumerate(ground_truth):
        query_key  = f"query_{entry['query_n']}"

        if query_key in results and query_key in already_done:
            print(f"[{idx+1}/{total}] {query_key} — already done, skipping")
            continue

        user_input = entry["user_input"]
        print(f"[{idx+1}/{total}] {query_key}: {user_input}")

        timings = {}

        # Query analysis
        with timer("query_analysis", timings):
            enriched, triggered = enrich_query(user_input)

        # ── Condition 1: RAW ──────────────────────────────────────────────────
        with timer("retrieval_raw", timings):
            raw_all = retrieve_all(user_input, top_k=top_k,
                                   timings=timings, prefix="raw_")

        with timer("generation_rrf_raw", timings):
            raw_ans = generate(user_input, raw_all["rrf"])

        raw_pages = [r["metadata"].get("page_number") for r in raw_all["rrf"]]

        # ── Condition 2: ENRICHED ─────────────────────────────────────────────
        with timer("retrieval_enriched", timings):
            enr_all = retrieve_all(enriched, top_k=top_k,
                                   timings=timings, prefix="enriched_")

        with timer("generation_rrf_enriched", timings):
            enr_ans = generate(enriched, enr_all["rrf"])

        enr_pages = [r["metadata"].get("page_number") for r in enr_all["rrf"]]

        # ── Condition 3: RERANKED ─────────────────────────────────────────────
        # Cast wider net for reranking — top_k * 2 from RRF
        with timer("retrieval_reranked", timings):
            enr_wide        = retrieve_all(enriched, top_k=top_k * 2,
                                           timings=timings, prefix="reranked_")
            reranked_chunks = rerank(enriched, enr_wide["rrf"])[:top_k]

        with timer("generation_reranked", timings):
            rer_ans = generate(enriched, reranked_chunks)

        rer_pages  = [r["metadata"].get("page_number") for r in reranked_chunks]
        rer_scores = [round(r.get("rerank_score", 0), 3) for r in reranked_chunks]

        # ── Store results ─────────────────────────────────────────────────────
        results[query_key] = {
            "user_input":     user_input,
            "category":       entry.get("category", ""),
            "relevant_pages": entry.get("relevant_pages", []),
            "enriched_query": enriched,
            "triggered":      triggered,
            "raw": {
                "retrieved_pages": raw_pages,
                "answer":          raw_ans,
            },
            "enriched": {
                "retrieved_pages": enr_pages,
                "answer":          enr_ans,
            },
            "reranked": {
                "retrieved_pages": rer_pages,
                "rerank_scores":   rer_scores,
                "answer":          rer_ans,
            },
        }

        latency_records.append({
            "query_key":  query_key,
            "user_input": user_input,
            "category":   entry.get("category", ""),
            **timings,
        })

        # Save incrementally after every query
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        with open(latency_path, "w", encoding="utf-8") as f:
            json.dump(latency_records, f, indent=2, ensure_ascii=False)

    print(f"\nDone — {len(results)}/{total} queries evaluated")
    print(f"Results  → {output_path}")
    print(f"Latency  → {latency_path}")
    return results, latency_records


# ── Latency summary ───────────────────────────────────────────────────────────

def summarise_latency(latency_records):
    """Print mean ± std per stage across all queries."""
    if not latency_records:
        print("No latency records found.")
        return

    # Collect all stage names (excluding metadata keys)
    meta_keys = {"query_key", "user_input", "category"}
    stages    = [k for k in latency_records[0].keys() if k not in meta_keys]

    print(f"\n{'Stage':<35} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 70)

    for stage in stages:
        vals = [r[stage] for r in latency_records if stage in r]
        if not vals:
            continue
        print(
            f"{stage:<35} "
            f"{np.mean(vals):>7.3f}s "
            f"{np.std(vals):>7.3f}s "
            f"{np.min(vals):>7.3f}s "
            f"{np.max(vals):>7.3f}s"
        )

    # Total per query
    total_per_query = [
        sum(r.get(s, 0) for s in stages) for r in latency_records
    ]
    print("-" * 70)
    print(
        f"{'TOTAL (per query)':<35} "
        f"{np.mean(total_per_query):>7.3f}s "
        f"{np.std(total_per_query):>7.3f}s "
        f"{np.min(total_per_query):>7.3f}s "
        f"{np.max(total_per_query):>7.3f}s"
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch evaluation")
    parser.add_argument("--ground_truth", required=True)
    parser.add_argument("--output",       required=True, help="Path for evaluation_results.json")
    parser.add_argument("--latency",      required=True, help="Path for latency_records.json")
    parser.add_argument("--vehicle",      required=True, help="Vehicle name (e.g. punto)")
    parser.add_argument("--top_k",        type=int, default=5)
    args = parser.parse_args()

    from online.pipeline import load_vehicle
    load_vehicle(args.vehicle)

    results, latency = run_evaluation(
        ground_truth_path = args.ground_truth,
        output_path       = args.output,
        latency_path      = args.latency,
        top_k             = args.top_k,
    )

    summarise_latency(latency)
