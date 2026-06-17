"""
indexing.py
-----------
Embeds vehicle manual chunks and stores them in a ChromaDB vector collection.

Reads chunks.json produced by chunking.py and builds the dense vector index
used for semantic retrieval at query time. The BM25 index is NOT built here
— it is rebuilt from chunks.json at session startup in the online pipeline
(rebuilding takes ~2 seconds and BM25 cannot be persisted).

Usage:
    python indexing.py \
        --chunks   path/to/chunks.json \
        --chroma   path/to/chroma_db/ \
        --vehicle  punto_2017
"""

import json
import argparse

import chromadb
from sentence_transformers import SentenceTransformer


# ── Constants ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL  = "all-mpnet-base-v2"
EMBEDDING_BATCH  = 64
CHROMA_SPACE     = "cosine"
# ─────────────────────────────────────────────────────────────────────────────


def load_chunks(chunks_path):
    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks from {chunks_path}")
    return chunks


def load_embedder(model_name=EMBEDDING_MODEL):
    """
    Load the sentence embedding model.

    all-mpnet-base-v2 was selected over the ChromaDB default (all-MiniLM-L6-v2)
    for its superior performance on technical domain text.
    It produces 768-dimensional dense vectors compared to 384 for MiniLM.
    The model is English-only, consistent with the English manual corpus.
    """
    print(f"Loading embedding model: {model_name}")
    embedder = SentenceTransformer(model_name)
    print("Embedding model ready")
    return embedder


def get_or_create_collection(chroma_path, vehicle_name):
    """
    Connect to ChromaDB and return the named collection, creating it if
    it does not exist.

    ChromaDB runs in fully embedded mode — no external server, no network
    calls. Data is persisted to chroma_path on disk.

    Each vehicle manual gets its own named collection, ensuring that queries
    retrieve exclusively from the relevant vehicle's knowledge base with no
    cross-vehicle knowledge bleed.

    Cosine similarity is used as the distance metric, consistent with the
    normalised dense embeddings produced by all-mpnet-base-v2.
    """
    client   = chromadb.PersistentClient(path=chroma_path)
    existing = [c.name for c in client.list_collections()]

    if vehicle_name in existing:
        collection = client.get_collection(vehicle_name)
        print(f"Loaded existing collection '{vehicle_name}' — {collection.count()} chunks")
        return collection, False  # False = not newly created

    collection = client.create_collection(
        name     = vehicle_name,
        metadata = {"hnsw:space": CHROMA_SPACE}
    )
    print(f"Created new collection '{vehicle_name}'")
    return collection, True  # True = newly created, needs indexing


def index_chunks(collection, chunks, embedder):
    """
    Embed all chunks and add them to the ChromaDB collection.

    Embeddings are computed in batches for efficiency. The same model
    (all-mpnet-base-v2) must be used at query time to ensure that query
    and document vectors lie in the same embedding space.
    """
    texts = [c["text"]      for c in chunks]
    ids   = [c["chunk_id"]  for c in chunks]
    metas = [c["metadata"]  for c in chunks]

    print(f"Embedding {len(texts)} chunks (batch size={EMBEDDING_BATCH})...")
    embeddings = embedder.encode(
        texts,
        show_progress_bar = True,
        batch_size        = EMBEDDING_BATCH,
    )

    collection.add(
        ids        = ids,
        documents  = texts,
        embeddings = embeddings.tolist(),
        metadatas  = metas,
    )
    print(f"Indexed {collection.count()} chunks into ChromaDB collection '{collection.name}'")


def run(chunks_path, chroma_path, vehicle_name):
    """
    Full indexing pipeline:
    1. Load chunks from JSON
    2. Load embedding model
    3. Get or create ChromaDB collection
    4. Embed and index chunks (skipped if collection already exists)
    """
    chunks                  = load_chunks(chunks_path)
    embedder                = load_embedder()
    collection, is_new      = get_or_create_collection(chroma_path, vehicle_name)

    if is_new:
        index_chunks(collection, chunks, embedder)
    else:
        print("Collection already indexed — skipping embedding step.")
        print("To re-index, delete the collection first or use --force.")

    return collection


# ── CLI entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Embed manual chunks and store in ChromaDB"
    )
    parser.add_argument("--chunks",  required=True, help="Path to chunks JSON file")
    parser.add_argument("--chroma",  required=True, help="Path to ChromaDB directory")
    parser.add_argument("--vehicle", required=True, help="Vehicle name (collection name)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing collection and re-index from scratch"
    )
    args = parser.parse_args()

    if args.force:
        client = chromadb.PersistentClient(path=args.chroma)
        existing = [c.name for c in client.list_collections()]
        if args.vehicle in existing:
            client.delete_collection(args.vehicle)
            print(f"Deleted existing collection '{args.vehicle}'")

    run(
        chunks_path  = args.chunks,
        chroma_path  = args.chroma,
        vehicle_name = args.vehicle,
    )
