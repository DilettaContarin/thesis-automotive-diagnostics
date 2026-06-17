"""
generation.py
-------------
LLM-based answer generation for the online pipeline.

Model: Mistral-7B-Instruct-v0.2 with 4-bit quantization (bitsandbytes nf4).

Design choices:
  - Greedy decoding (do_sample=False): deterministic output — identical queries
    always produce identical answers, essential for evaluation reproducibility
    and consistency in a safety-critical diagnostic context.
  - 4-bit quantization: enables Mistral-7B to fit within T4 GPU VRAM (15GB)
    while preserving generation quality.
  - Constrained prompt: model is explicitly instructed to use only retrieved
    excerpts and to cite page numbers. Hallucination cutoff removes any
    continuation beyond the first answer (Mistral occasionally generates
    additional invented Q&A pairs due to training data patterns).
  - Page-level citations: every retrieved chunk is labelled with its source
    page, enabling users to verify answers against the original manual.

GPU memory note:
  Mistral-7B and Whisper cannot coexist in T4 VRAM simultaneously.
  Load models sequentially: run ASR first, then free Whisper before loading
  Mistral, or load Mistral first and handle ASR in a separate session.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_ID       = "mistralai/Mistral-7B-Instruct-v0.2"
MAX_NEW_TOKENS = 300
# ─────────────────────────────────────────────────────────────────────────────

# ── Model instances (loaded once) ─────────────────────────────────────────────
_tokenizer = None
_model     = None


def load_model():
    """
    Load Mistral-7B-Instruct-v0.2 with 4-bit quantization.
    Idempotent — safe to call multiple times.
    """
    global _tokenizer, _model
    if _model is not None:
        return _tokenizer, _model

    bnb_config = BitsAndBytesConfig(
        load_in_4bit              = True,
        bnb_4bit_use_double_quant = True,
        bnb_4bit_quant_type       = "nf4",
        bnb_4bit_compute_dtype    = torch.float16,
    )

    print("Loading tokenizer...")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("Loading Mistral-7B (first time ~10 min download, subsequent ~1 min)...")
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config = bnb_config,
        device_map          = "auto",
    )
    print(f"Mistral loaded on: {next(_model.parameters()).device}")
    return _tokenizer, _model


# ── Prompt construction ───────────────────────────────────────────────────────

def build_prompt(query, chunks_for_generation):
    """
    Assemble the RAG prompt from retrieved chunks and the user query.

    Each chunk is labelled with its source page to enable the model to produce
    page-level citations (e.g. [Page 47]). The prompt strictly constrains the
    model to retrieved content and requires abstention ("I could not find this
    in the manual") when the answer is not present, preventing hallucination.
    """
    context_parts = []
    for i, chunk in enumerate(chunks_for_generation):
        page = chunk["metadata"].get("page_number", "?")
        context_parts.append(f"[Chunk {i+1} | Page {page}]\n{chunk['text']}")

    context = "\n\n".join(context_parts)

    return f"""You are an automotive diagnostic assistant. Answer the technician's question using ONLY the manual excerpts provided below.

Rules:
- Only use information from the provided excerpts
- Always cite the page number in square brackets e.g. [Page 47]
- If the answer is not in the excerpts, say "I could not find this in the manual"
- Be concise and precise

Manual excerpts:
{context}

Technician's question: {query}

Answer:"""


# ── Generation ────────────────────────────────────────────────────────────────

def generate(query, chunks_for_generation, max_new_tokens=MAX_NEW_TOKENS):
    """
    Generate a grounded diagnostic answer from retrieved chunks.

    Args:
        query:                   the user query (enriched or raw)
        chunks_for_generation:   list of result dicts from retrieval/reranking
        max_new_tokens:          maximum tokens to generate

    Returns:
        answer (str): the generated answer with page citations
    """
    if not chunks_for_generation:
        return "I could not find relevant information in the manual."

    tokenizer, model = load_model()
    prompt           = build_prompt(query, chunks_for_generation)
    inputs           = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len        = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens = max_new_tokens,
            do_sample      = False,     # greedy decoding — deterministic output
            temperature    = 1.0,       # no effect when do_sample=False
            pad_token_id   = tokenizer.eos_token_id,
            eos_token_id   = tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][input_len:]
    answer     = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Hallucination cutoff: Mistral occasionally continues generating
    # additional invented Q&A pairs due to training data patterns.
    if "Technician's question:" in answer:
        answer = answer[:answer.index("Technician's question:")].strip()

    return answer.strip()
