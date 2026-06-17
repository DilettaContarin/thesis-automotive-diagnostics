"""
icon_extraction.py
------------------
Extracts, normalises, and deduplicates SVG icons from a vehicle manual PDF.
Outputs one PNG + one JSON sidecar file per unique icon to the output directory.

Usage:
    python icon_extraction.py \
        --pdf path/to/manual.pdf \
        --output path/to/output_dir
"""

import os
import io
import json
import argparse
from pathlib import Path

import fitz
import imagehash
from PIL import Image


# ── Constants (edit here or override via CLI) ─────────────────────────────────
DEFAULT_DPI              = 288
FIRST_PASS_MIN_PATHS     = 8
SECOND_PASS_MIN_PATHS    = 3
SECOND_PASS_MAX_PATHS    = 7
NORMALISE_TARGET_SIZE    = 128
DEDUP_HASH_THRESHOLD     = 10
# ─────────────────────────────────────────────────────────────────────────────


def extract_icons(doc, dpi=DEFAULT_DPI,
                  first_pass_min=FIRST_PASS_MIN_PATHS,
                  second_pass_min=SECOND_PASS_MIN_PATHS,
                  second_pass_max=SECOND_PASS_MAX_PATHS):
    """
    Extract icon candidates from all pages of a PyMuPDF document.

    Two-pass strategy:
      - First pass:  icons with >= first_pass_min path operations (complex icons)
      - Second pass: icons with second_pass_min <= n_paths < first_pass_min (simple symbols)

    Returns a list of dicts with keys: img, page, index, n_paths, pass, rect.
    Coordinates in rect are in original PDF point space (bottom-left origin).
    """
    mat   = fitz.Matrix(dpi / 72, dpi / 72)
    icons = []

    for page_num, page in enumerate(doc):
        drawings = page.get_drawings()
        for i, drawing in enumerate(drawings):
            rect = drawing["rect"]
            if rect.is_empty or rect.is_infinite:
                continue

            n_paths = len(drawing["items"])
            w, h    = rect.width, rect.height
            if w < 5 or h < 5:
                continue

            is_first  = n_paths >= first_pass_min
            is_second = second_pass_min <= n_paths < first_pass_min
            if not (is_first or is_second):
                continue

            clip = rect + fitz.Rect(-2, -2, 2, 2)
            pix  = page.get_pixmap(matrix=mat, clip=clip)
            img  = Image.open(io.BytesIO(pix.tobytes("png")))

            icons.append({
                "img":    img,
                "page":   page_num + 1,
                "index":  i,
                "n_paths": n_paths,
                "pass":   "first" if is_first else "second",
                "rect":   [rect.x0, rect.y0, rect.x1, rect.y1],
            })

    first  = sum(1 for x in icons if x["pass"] == "first")
    second = sum(1 for x in icons if x["pass"] == "second")
    print(f"Extracted {len(icons)} candidates ({first} first pass, {second} second pass)")
    return icons


def normalise_icons(icons, target_size=NORMALISE_TARGET_SIZE):
    """
    Normalise extracted icon images:
      - Flatten RGBA to white background
      - Convert to greyscale
      - Binarize at threshold 128 (critical for reliable phash matching)
      - Resize to target_size × target_size square with white padding

    Adds 'img_normalized' key to each icon dict in place.
    """
    for item in icons:
        img = item["img"].convert("RGBA")
        bg  = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg.convert("L")
        img = img.point(lambda x: 0 if x < 128 else 255, "1").convert("RGB")
        img.thumbnail((target_size, target_size), Image.LANCZOS)
        square = Image.new("RGB", (target_size, target_size), (255, 255, 255))
        offset = ((target_size - img.width) // 2, (target_size - img.height) // 2)
        square.paste(img, offset)
        item["img_normalized"] = square

    print(f"Normalised {len(icons)} icons")
    return icons


def deduplicate_icons(icons, hash_threshold=DEDUP_HASH_THRESHOLD):
    """
    Group visually identical icons using perceptual hashing (phash).
    Icons whose phash distance is <= hash_threshold are treated as the same icon.

    Returns a list of unique icon entries, each with:
      - img_normalized: the representative normalised image
      - phash:          string representation of the perceptual hash
      - occurrences:    list of {page, rect, n_paths} dicts
    """
    seen   = {}
    unique = []

    for item in icons:
        h       = imagehash.phash(item["img_normalized"])
        matched = None

        for seen_hash, uid in seen.items():
            if abs(h - seen_hash) <= hash_threshold:
                matched = uid
                break

        occurrence = {
            "page":    item["page"],
            "rect":    item["rect"],
            "n_paths": item["n_paths"],
        }

        if matched is None:
            seen[h] = len(unique)
            unique.append({
                "img_normalized": item["img_normalized"],
                "phash":          str(h),
                "occurrences":    [occurrence],
            })
        else:
            unique[matched]["occurrences"].append(occurrence)

    total_occurrences = sum(len(u["occurrences"]) for u in unique)
    print(f"Deduplication done:")
    print(f"  Unique icons:      {len(unique)}")
    print(f"  Total occurrences: {total_occurrences}")
    return unique


def save_icons(unique_icons, output_dir):
    """
    Save each unique icon as:
      - icon_NNN.png  — the normalised representative image
      - icon_NNN.json — sidecar with phash, occurrences, and PDF coordinates

    The description field is left empty for manual annotation.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, item in enumerate(unique_icons):
        base_name = f"icon_{i+1:03d}"

        # Save image
        item["img_normalized"].save(output_dir / f"{base_name}.png")

        # Save sidecar JSON
        sidecar = {
            "name":          base_name,
            "description":   "",           # filled manually after extraction
            "phash":         item["phash"],
            "n_occurrences": len(item["occurrences"]),
            "occurrences":   item["occurrences"],
        }
        with open(output_dir / f"{base_name}.json", "w") as f:
            json.dump(sidecar, f, indent=2)

    print(f"Saved {len(unique_icons)} unique icons to {output_dir}")


def run(pdf_path, output_dir):
    """Full extraction pipeline: open PDF → extract → normalise → dedup → save."""
    doc = fitz.open(pdf_path)
    print(f"Loaded: {len(doc)} pages from {pdf_path}")

    raw        = extract_icons(doc)
    normalised = normalise_icons(raw)
    unique     = deduplicate_icons(normalised)
    save_icons(unique, output_dir)

    doc.close()
    return unique


# ── CLI entrypoint ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract icons from a vehicle manual PDF")
    parser.add_argument("--pdf",    required=True, help="Path to the PDF manual")
    parser.add_argument("--output", required=True, help="Output directory for icons")
    args = parser.parse_args()

    run(pdf_path=args.pdf, output_dir=args.output)
