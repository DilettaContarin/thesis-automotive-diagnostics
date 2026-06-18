"""
chunking.py
-----------
Parses a vehicle manual PDF, injects icon descriptions into text elements,
and produces structure-aware chunks saved to a JSON file.

Requires icon sidecar JSON files produced by icon_extraction.py (with manual
descriptions filled in).

Usage:
    python chunking.py \
        --pdf      path/to/manual.pdf \
        --icons    path/to/icons_dir/ \
        --output   path/to/chunks.json \
        --vehicle  punto
"""

import json
import glob
import argparse
from pathlib import Path
from collections import defaultdict

import fitz
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title


# ── Constants ─────────────────────────────────────────────────────────────────
MAX_DISTANCE_PT         = 120   # maximum spatial distance for icon→element matching
CHUNK_MAX_CHARACTERS    = 500
CHUNK_COMBINE_UNDER     = 100
# ─────────────────────────────────────────────────────────────────────────────


# ── Coordinate utilities ──────────────────────────────────────────────────────

def get_page_heights(pdf_path):
    """
    Return {page_number: height_in_points} for all pages.

    Needed because partition_pdf (pdfminer) places the origin at the
    top-left corner, while PyMuPDF get_drawings() uses the bottom-left
    corner. Page height is required to reconcile the two systems.
    """
    doc     = fitz.open(pdf_path)
    heights = {i + 1: page.rect.height for i, page in enumerate(doc)}
    doc.close()
    return heights


def get_element_bbox(el):
    """
    Return (x0, y0, x1, y1) in pdfminer coordinate space (bottom-left origin).
    Returns None if the element has no coordinate metadata.
    """
    coords = el.metadata.coordinates
    if coords is None:
        return None
    pts = coords.points
    xs  = [p[0] for p in pts]
    ys  = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def icon_center_pdfminer(rect, page_height):
    """
    Convert a PyMuPDF icon rect (top-left origin) to its center point
    expressed in pdfminer coordinate space (bottom-left origin).
    """
    x0, y0, x1, y1 = rect
    cx = (x0 + x1) / 2
    cy = page_height - (y0 + y1) / 2
    return cx, cy


def bbox_contains(bbox, cx, cy):
    x0, y0, x1, y1 = bbox
    return x0 <= cx <= x1 and y0 <= cy <= y1


def edge_distance(bbox, cx, cy):
    """
    Distance from point (cx, cy) to the nearest point on bbox edge.
    Returns 0 if the point is inside the bbox.

    Preferred over center-to-center distance for large text blocks where
    an icon may be visually adjacent to the edge yet far from the center.
    """
    x0, y0, x1, y1 = bbox
    dx = max(x0 - cx, 0, cx - x1)
    dy = max(y0 - cy, 0, cy - y1)
    return (dx ** 2 + dy ** 2) ** 0.5


# ── Icon loading ──────────────────────────────────────────────────────────────

def load_icons(icons_dir):
    """
    Load all icon sidecar JSON files from icons_dir.
    Each file must have at minimum: description, occurrences[{page, rect}].
    """
    icons = []
    for path in glob.glob(str(Path(icons_dir) / "*.json")):
        with open(path, encoding="utf-8") as f:
            icons.append(json.load(f))
    print(f"Loaded {len(icons)} icon definitions from {icons_dir}")
    return icons


# ── PDF partitioning ──────────────────────────────────────────────────────────

def partition_manual(pdf_path):
    """
    Partition the PDF into text elements using pdfminer (strategy='fast').

    strategy='fast' is used rather than 'hi_res' because:
    - It preserves element bounding box coordinates in PDF point space,
      which are required for icon spatial matching.
    - It is significantly faster for large manuals.
    Known limitation: table row/column structure is not reconstructed;
    cells are extracted as flat text. For automotive warning light tables
    this is acceptable as each cell contains self-contained text.
    """
    print(f"Partitioning {pdf_path} (strategy=fast)...")
    elements = partition_pdf(pdf_path, strategy="fast")
    print(f"Total elements: {len(elements)}")
    return elements


# ── Icon injection ────────────────────────────────────────────────────────────

def inject_icons(elements, icons, page_heights, max_distance=MAX_DISTANCE_PT):
    """
    For each icon occurrence, find the nearest text element on the same page
    and prepend the icon description to that element's text as [ICON: ...].

    Matching strategy:
    1. If the icon center falls inside an element's bounding box → immediate match
    2. Otherwise find the element with the minimum edge distance

    An occurrence is skipped if:
    - The page has no elements
    - The nearest element is farther than max_distance points away

    Returns (injected_count, skipped_count).
    """
    # Index elements by page for fast lookup
    elements_by_page = defaultdict(list)
    for el in elements:
        page = el.metadata.page_number
        if page is not None:
            elements_by_page[page].append(el)

    injected = 0
    skipped  = 0

    for icon in icons:
        description = icon.get("description", "")
        if not description:
            continue  # skip unannotated icons

        for occ in icon["occurrences"]:
            page   = occ["page"]
            rect   = occ["rect"]
            page_h = page_heights.get(page)

            if page_h is None:
                skipped += 1
                continue

            cx, cy     = icon_center_pdfminer(rect, page_h)
            candidates = elements_by_page.get(page, [])

            if not candidates:
                skipped += 1
                continue

            best_el   = None
            best_dist = float("inf")

            for el in candidates:
                bbox = get_element_bbox(el)
                if bbox is None:
                    continue
                if bbox_contains(bbox, cx, cy):
                    best_el   = el
                    best_dist = 0
                    break
                dist = edge_distance(bbox, cx, cy)
                if dist < best_dist:
                    best_dist = dist
                    best_el   = el

            if best_el is None or best_dist > max_distance:
                skipped += 1
                continue

            best_el.text = f"[ICON: {description}] " + best_el.text
            injected += 1

    print(f"Icon injection complete — injected: {injected} | skipped: {skipped}")
    return injected, skipped


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_elements(elements,
                   max_characters=CHUNK_MAX_CHARACTERS,
                   combine_under=CHUNK_COMBINE_UNDER):
    """
    Group elements into structure-aware chunks using unstructured.io chunk_by_title.

    chunk_by_title respects document section headers, keeping related content
    together rather than splitting at arbitrary character counts — well-suited
    to the hierarchical structure of vehicle service manuals.
    """
    chunks = chunk_by_title(
        elements,
        max_characters=max_characters,
        combine_text_under_n_chars=combine_under,
    )
    print(f"Chunking complete — total chunks: {len(chunks)}")
    return chunks


# ── Serialisation ─────────────────────────────────────────────────────────────

def chunks_to_json(chunks, vehicle_name):
    """
    Convert chunk objects to serialisable dicts.

    Each dict contains:
    - chunk_id:    unique identifier scoped to the vehicle
    - text:        chunk text content (may include [ICON: ...] prefixes)
    - metadata:
        - page_number:  source page in the PDF (used for citations)
        - chunk_index:  sequential index (used by BM25 retriever)
    """
    output = []
    for i, chunk in enumerate(chunks):
        output.append({
            "chunk_id": f"{vehicle_name}_chunk_{i:04d}",
            "text":     chunk.text,
            "metadata": {
                "page_number": chunk.metadata.page_number,
                "chunk_index": i,
            }
        })
    return output


def save_chunks(chunks_data, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(chunks_data)} chunks → {output_path}")


# ── Diagnostics ───────────────────────────────────────────────────────────────

def print_icon_chunk_summary(chunks_data, n=3):
    """Print a summary of chunks containing icon descriptions."""
    icon_chunks = [c for c in chunks_data if "[ICON:" in c["text"]]
    print(f"\nChunks containing icon descriptions: {len(icon_chunks)}")
    for c in icon_chunks[:n]:
        page = c["metadata"].get("page_number", "?")
        print(f"  Page {page} | {c['text'][:200]}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(pdf_path, icons_dir, output_path, vehicle_name):
    """
    Full offline chunking pipeline:
    1. Get page heights for coordinate reconciliation
    2. Load icon sidecar JSONs
    3. Partition PDF into text elements
    4. Inject icon descriptions into nearest elements
    5. Chunk elements with structure-aware splitting
    6. Save to JSON
    """
    page_heights = get_page_heights(pdf_path)
    icons        = load_icons(icons_dir)
    elements     = partition_manual(pdf_path)
    inject_icons(elements, icons, page_heights)
    chunks       = chunk_elements(elements)
    chunks_data  = chunks_to_json(chunks, vehicle_name)
    save_chunks(chunks_data, output_path)
    print_icon_chunk_summary(chunks_data)
    return chunks_data


# ── CLI entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse a vehicle manual PDF and produce icon-injected chunks"
    )
    parser.add_argument("--pdf",      required=True, help="Path to the PDF manual")
    parser.add_argument("--icons",    required=True, help="Directory containing icon JSON sidecars")
    parser.add_argument("--output",   required=True, help="Output path for chunks JSON")
    parser.add_argument("--vehicle",  required=True, help="Vehicle name (e.g. punto_2017)")
    args = parser.parse_args()

    run(
        pdf_path     = args.pdf,
        icons_dir    = args.icons,
        output_path  = args.output,
        vehicle_name = args.vehicle,
    )
