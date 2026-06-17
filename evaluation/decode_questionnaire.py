import json
import numpy as np
from collections import defaultdict

# ── Paths  ───────────────────────────────────────────────────────────────────
MAPPING_PATH     = "/content/drive/MyDrive/label_mapping.json"
EVALUATOR_1_PATH = "/content/drive/MyDrive/evaluator_1_responses.json"
EVALUATOR_2_PATH = "/content/drive/MyDrive/evaluator_2_responses.json"
DECODED_PATH     = "/content/drive/MyDrive/decoded_results.json"
# ─────────────────────────────────────────────────────────────────────────────
# Expected format for evaluator response files:
# [
#   {
#     "query_key": "query_1",
#     "ranking": {"A": 1, "B": 3, "C": 2},
#     "relevance": {"A": 4, "B": 3, "C": 5},
#     "accuracy": {"A": 4, "B": 3, "C": 5},
#     "citation": true
#   },
#   ...
# ]
# ─────────────────────────────────────────────────────────────────────────────

with open(MAPPING_PATH, encoding="utf-8") as f:
    label_map = json.load(f)

with open(EVALUATOR_1_PATH, encoding="utf-8") as f:
    eval1 = {e["query_key"]: e for e in json.load(f)}

with open(EVALUATOR_2_PATH, encoding="utf-8") as f:
    eval2 = {e["query_key"]: e for e in json.load(f)}


def decode_scores(evaluator_data, label_map):
    """Convert letter-based scores to condition-based scores."""
    decoded = {}
    for query_key, response in evaluator_data.items():
        mapping = label_map[query_key]          # e.g. {"A": "raw", "B": "reranked", "C": "enriched"}
        reverse = {v: k for k, v in mapping.items()}  # e.g. {"raw": "A", ...}

        decoded[query_key] = {
            "ranking":   {cond: response["ranking"][reverse[cond]]   for cond in ["raw", "enriched", "reranked"]},
            "relevance": {cond: response["relevance"][reverse[cond]] for cond in ["raw", "enriched", "reranked"]},
            "accuracy":  {cond: response["accuracy"][reverse[cond]]  for cond in ["raw", "enriched", "reranked"]},
            "citation":  response.get("citation", None),
        }
    return decoded


decoded1 = decode_scores(eval1, label_map)
decoded2 = decode_scores(eval2, label_map)

# Save decoded results
with open(DECODED_PATH, "w", encoding="utf-8") as f:
    json.dump({"evaluator_1": decoded1, "evaluator_2": decoded2},
              f, indent=2, ensure_ascii=False)

print(f"Decoded results saved → {DECODED_PATH}")


# ── Summary statistics ────────────────────────────────────────────────────────
def summarise(decoded1, decoded2, metric):
    conditions = ["raw", "enriched", "reranked"]
    print(f"\n{metric.upper()} — mean ± std (lower is better for ranking, higher for relevance/accuracy)")
    print(f"{'Condition':<12} {'Eval 1':>10} {'Eval 2':>10} {'Combined':>12}")
    print("-" * 48)
    for cond in conditions:
        scores1 = [v[metric][cond] for v in decoded1.values() if metric in v]
        scores2 = [v[metric][cond] for v in decoded2.values() if metric in v]
        combined = scores1 + scores2
        print(
            f"  {cond:<10} "
            f"{np.mean(scores1):>6.2f}±{np.std(scores1):.2f}  "
            f"{np.mean(scores2):>6.2f}±{np.std(scores2):.2f}  "
            f"{np.mean(combined):>8.2f}±{np.std(combined):.2f}"
        )

summarise(decoded1, decoded2, "ranking")
summarise(decoded1, decoded2, "relevance")
summarise(decoded1, decoded2, "accuracy")


# ── Inter-annotator agreement on ranking ─────────────────────────────────────
from scipy.stats import spearmanr

print("\nINTER-ANNOTATOR AGREEMENT (Spearman correlation on rankings):")
for cond in ["raw", "enriched", "reranked"]:
    ranks1 = [decoded1[k]["ranking"][cond] for k in decoded1 if k in decoded2]
    ranks2 = [decoded2[k]["ranking"][cond] for k in decoded1 if k in decoded2]
    corr, pval = spearmanr(ranks1, ranks2)
    print(f"  {cond:<12} r = {corr:.3f}  (p = {pval:.3f})")


# ── Per-category breakdown ────────────────────────────────────────────────────
# Load form_ready to get categories
form_ready_path = "/content/drive/MyDrive/form_ready.json"
with open(form_ready_path, encoding="utf-8") as f:
    form_entries = {e["query_key"]: e["category"] for e in json.load(f)}

print("\nRANKING BY CATEGORY (mean rank, lower = better):")
categories = ["icon", "symptom", "technical", "component"]
conditions = ["raw", "enriched", "reranked"]

for cat in categories:
    cat_keys = [k for k, v in form_entries.items() if v == cat]
    print(f"\n  {cat.upper()} ({len(cat_keys)} queries):")
    for cond in conditions:
        scores = []
        for k in cat_keys:
            if k in decoded1:
                scores.append(decoded1[k]["ranking"][cond])
            if k in decoded2:
                scores.append(decoded2[k]["ranking"][cond])
        if scores:
            print(f"    {cond:<12} mean rank = {np.mean(scores):.2f}")


# ── Citation rate ─────────────────────────────────────────────────────────────
cit1 = [v["citation"] for v in decoded1.values() if v["citation"] is not None]
cit2 = [v["citation"] for v in decoded2.values() if v["citation"] is not None]
combined_cit = cit1 + cit2
print(f"\nCITATION RATE: {sum(combined_cit)}/{len(combined_cit)} "
      f"({100*sum(combined_cit)/len(combined_cit):.1f}% of responses cited a page number)")
