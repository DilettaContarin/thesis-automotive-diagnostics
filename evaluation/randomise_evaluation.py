import json
import random
import os

# ── Paths (edit these) ────────────────────────────────────────────────────────
EVAL_RESULTS_PATH = "/content/drive/MyDrive/evaluation_results.json"
OUTPUT_FORM_PATH  = "/content/drive/MyDrive/form_ready.json"
MAPPING_PATH      = "/content/drive/MyDrive/label_mapping.json"  
# ─────────────────────────────────────────────────────────────────────────────

random.seed(42)  

with open(EVAL_RESULTS_PATH, encoding="utf-8") as f:
    results = json.load(f)

form_entries = []   # what evaluators see
label_map    = {}   # private mapping: which letter = which condition

for query_key, entry in results.items():

    # --- Build the three responses ---
    conditions = {
        "raw":      entry["raw"]["answer"],
        "enriched": entry["enriched"]["answer"],
        "reranked": entry["reranked"]["answer"],
    }

    # Randomise which letter (A/B/C) maps to which condition
    letters    = ["A", "B", "C"]
    cond_names = list(conditions.keys())
    random.shuffle(cond_names)

    assignment = {letter: cond for letter, cond in zip(letters, cond_names)}

    # Store private mapping
    label_map[query_key] = {
        letter: cond for letter, cond in assignment.items()
    }

    # Build form entry
    form_entry = {
        "query_key":  query_key,
        "user_input": entry["user_input"],
        "category":   entry.get("category", ""),
        "responses": {
            "A": conditions[assignment["A"]],
            "B": conditions[assignment["B"]],
            "C": conditions[assignment["C"]],
        }
    }
    form_entries.append(form_entry)

# Save form-ready file
with open(OUTPUT_FORM_PATH, "w", encoding="utf-8") as f:
    json.dump(form_entries, f, indent=2, ensure_ascii=False)

# Save private mapping — DO NOT share with evaluators
with open(MAPPING_PATH, "w", encoding="utf-8") as f:
    json.dump(label_map, f, indent=2, ensure_ascii=False)

print(f"Form-ready file saved → {OUTPUT_FORM_PATH}")
print(f"Label mapping saved   → {MAPPING_PATH}  (keep private!)")
print(f"Total queries:          {len(form_entries)}")

# Quick sanity check
entry = form_entries[0]
print(f"\nSample — {entry['query_key']}: {entry['user_input']}")
print(f"  A = {label_map[entry['query_key']]['A']}")
print(f"  B = {label_map[entry['query_key']]['B']}")
print(f"  C = {label_map[entry['query_key']]['C']}")
print(f"  Response A preview: {entry['responses']['A'][:80]}...")
