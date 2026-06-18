"""
metrics.py
----------
Retrieval quality metrics for the automotive diagnostics RAG pipeline.

Computes MRR, Recall@k and Precision@k across three ablation conditions:
  - raw:      baseline hybrid retrieval, no query enrichment
  - enriched: retrieval with vocabulary enrichment (Enhancement 1)
  - reranked: retrieval with enrichment + cross-encoder reranking (Enhancement 2)

Metrics are computed at query level and aggregated as means across all queries
and per category (icon, symptom, technical, component).

Usage:
    python evaluation/metrics.py \
        --ground_truth  data/ground_truth.json \
        --results       data/evaluation_results.json \
        --output        data/retrieval_metrics.csv
"""

import json
import argparse
import numpy as np
import pandas as pd


# ── Metric functions ──────────────────────────────────────────────────────────

def reciprocal_rank(retrieved_pages, relevant_pages):
    """
    Reciprocal rank of the first relevant page in the retrieved list.
    Returns 1/rank if a relevant page is found, 0 otherwise.
    """
    for rank, page in enumerate(retrieved_pages, start=1):
        if page in relevant_pages:
            return 1.0 / rank
    return 0.0


def recall_at_k(retrieved_pages, relevant_pages, k):
    """
    Fraction of relevant pages found in the top-k retrieved results.
    Measures coverage — did we find what we needed?
    """
    top_k = retrieved_pages[:k]
    hits  = sum(1 for p in relevant_pages if p in top_k)
    return hits / len(relevant_pages) if relevant_pages else 0.0


def precision_at_k(retrieved_pages, relevant_pages, k):
    """
    Fraction of top-k retrieved pages that are relevant.
    Measures precision — was what we retrieved useful?
    """
    top_k = retrieved_pages[:k]
    hits  = sum(1 for p in top_k if p in relevant_pages)
    return hits / k if k > 0 else 0.0


# ── Score computation ─────────────────────────────────────────────────────────

def compute_scores(ground_truth, results, k_values=(1, 3, 5)):
    """
    Compute per-query metrics for all three conditions.

    Args:
        ground_truth: list of dicts with query_n, relevant_pages, category
        results:      dict keyed by query_N with raw/enriched/reranked sub-dicts
        k_values:     tuple of k values for Recall@k and Precision@k

    Returns:
        scores: dict {condition: [row_dict, ...]} with per-query metric rows
    """
    conditions = ["raw", "enriched", "reranked"]
    scores     = {cond: [] for cond in conditions}

    for item in ground_truth:
        qkey           = f"query_{item['query_n']}"
        relevant_pages = set(item["relevant_pages"])

        if qkey not in results:
            print(f"Warning: {qkey} not in results — skipping")
            continue

        res = results[qkey]

        for cond in conditions:
            if cond not in res:
                continue
            retrieved = res[cond]["retrieved_pages"]

            row = {
                "query_n":  item["query_n"],
                "category": item["category"],
                "rr":       reciprocal_rank(retrieved, relevant_pages),
            }
            for k in k_values:
                row[f"recall@{k}"]    = recall_at_k(retrieved, relevant_pages, k)
                row[f"precision@{k}"] = precision_at_k(retrieved, relevant_pages, k)

            scores[cond].append(row)

    return scores


# ── Summary printing ──────────────────────────────────────────────────────────

def print_summary(scores, k_values=(1, 3, 5)):
    """Print aggregate metrics per condition."""
    conditions = ["raw", "enriched", "reranked"]

    for cond in conditions:
        score_list = scores[cond]
        print(f"\n")
        print(f"  {cond.upper()}")
        print(f"\n")
        print(f"  MRR:           {np.mean([s['rr'] for s in score_list]):.4f}")
        for k in k_values:
            rec  = np.mean([s[f"recall@{k}"]    for s in score_list])
            prec = np.mean([s[f"precision@{k}"] for s in score_list])
            print(f"  Recall@{k}:      {rec:.4f}   Precision@{k}: {prec:.4f}")


def print_category_breakdown(scores, k_values=(1, 3, 5)):
    """Print MRR and Recall@5 per category per condition."""
    conditions = ["raw", "enriched", "reranked"]
    categories = sorted(set(s["category"] for sl in scores.values() for s in sl))

    for cond in conditions:
        print(f"\n")
        print(f"  {cond.upper()} — by category")
        print(f"\n")
        for cat in categories:
            cat_scores = [s for s in scores[cond] if s["category"] == cat]
            if not cat_scores:
                continue
            mrr  = np.mean([s["rr"]          for s in cat_scores])
            rec5 = np.mean([s["recall@5"]    for s in cat_scores])
            print(f"  {cat:<15} n={len(cat_scores):<3}  MRR={mrr:.3f}   Recall@5={rec5:.3f}")


# ── Export ────────────────────────────────────────────────────────────────────

def save_csv(scores, output_path):
    """Save per-query scores for all conditions to CSV."""
    conditions = ["raw", "enriched", "reranked"]
    rows       = []
    for cond in conditions:
        for s in scores[cond]:
            rows.append({"condition": cond, **s})

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\nSaved per-query scores → {output_path}")

    # Quick summary table
    metric_cols = [c for c in df.columns if c not in ("condition", "query_n", "category")]
    print(df.groupby("condition")[metric_cols].mean().round(4).to_string())
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def run(ground_truth_path, results_path, output_path=None, k_values=(1, 3, 5)):
    with open(ground_truth_path, encoding="utf-8") as f:
        ground_truth = json.load(f)
    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)

    print(f"Ground truth: {len(ground_truth)} queries")
    print(f"Results:      {len(results)} queries")

    scores = compute_scores(ground_truth, results, k_values=k_values)
    print_summary(scores, k_values=k_values)
    print_category_breakdown(scores, k_values=k_values)

    if output_path:
        save_csv(scores, output_path)

    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute retrieval metrics")
    parser.add_argument("--ground_truth", required=True)
    parser.add_argument("--results",      required=True)
    parser.add_argument("--output",       default=None, help="Optional CSV output path")
    args = parser.parse_args()

    run(
        ground_truth_path = args.ground_truth,
        results_path      = args.results,
        output_path       = args.output,
    )
