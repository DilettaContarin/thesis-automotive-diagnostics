"""
query_analysis.py
-----------------
Text cleaning and vocabulary enrichment for the online pipeline.

Preprocessing steps:
  1. Apostrophe normalisation — converts curly apostrophes to straight ones.
     Critical: speech-to-text output uses curly apostrophes (') which do not
     match the straight apostrophes (') in SYNONYM_MAP keys.
  2. Text cleaning — lowercase, filler word removal, whitespace collapse,
     trailing punctuation stripping.
  3. Vocabulary enrichment — appends technical manual terms for any colloquial
     expressions found in the cleaned query (append-only, original preserved).
"""

import re
from synonym_map import SYNONYM_MAP


def normalize_apostrophes(text):
    """Convert curly/smart apostrophes to straight ones."""
    return text.replace('\u2019', "'").replace('\u2018', "'")


def clean_query(text):
    """
    Clean and normalise a raw query string.

    Operations:
    - Apostrophe normalisation (critical for SYNONYM_MAP matching)
    - Lowercase
    - Filler word removal (common in spoken ASR output)
    - Whitespace collapse
    - Trailing punctuation removal (prevents enrichment concatenation artefacts
      such as "my car won't start. engine starting failure")
    """
    text    = normalize_apostrophes(text.lower().strip())
    fillers = ["um", "uh", "like", "you know", "i mean", "so", "basically"]
    for f in fillers:
        text = re.sub(rf'\b{f}\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.rstrip('.,!?')
    return text


def enrich_query(query):
    """
    Expand a cleaned query with technical synonyms from SYNONYM_MAP.

    Mechanism: scans the cleaned query for any colloquial key in SYNONYM_MAP.
    When a match is found, the corresponding technical terms are appended to
    the end of the original query string (append-only — original is preserved).

    The modified variant of RRF used (k=0) makes top-ranked results from each
    retriever highly influential; enrichment ensures that both BM25 (exact term
    matching) and dense retrieval (semantic) benefit from the expanded vocabulary.

    Returns:
        enriched (str): the cleaned query, possibly extended with technical terms
        matched  (list[str]): the colloquial keys that triggered enrichment
    """
    cleaned    = clean_query(query)
    expansions = []
    matched    = []

    for colloquial, technical_terms in SYNONYM_MAP.items():
        if normalize_apostrophes(colloquial) in cleaned:
            expansions.extend(technical_terms)
            matched.append(colloquial)

    if expansions:
        # Deduplicate while preserving order
        seen     = set()
        unique   = [x for x in expansions if not (x in seen or seen.add(x))]
        enriched = cleaned + " " + " ".join(unique)
    else:
        enriched = cleaned

    return enriched, matched
