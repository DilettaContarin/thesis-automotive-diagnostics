"""
asr.py
------
Automatic Speech Recognition using OpenAI Whisper.

Whisper small (244M params) is used over medium (769M) for its lower
VRAM footprint on T4 GPU, while still producing transcriptions of
sufficient quality for the downstream retrieval and generation stages.

Runtime prompting with automotive vocabulary biases transcription toward
technical terms without requiring fine-tuning or labelled data.
"""

import whisper

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_SIZE     = "small"
LANGUAGE       = "en"
INITIAL_PROMPT = "Automotive diagnostic report. Car problems, warning lights, engine issues."
# ─────────────────────────────────────────────────────────────────────────────

_model_asr = None


def get_model(device="cuda"):
    """Load Whisper once and reuse. Idempotent — safe to call multiple times."""
    global _model_asr
    if _model_asr is None:
        print(f"Loading Whisper {MODEL_SIZE}...")
        _model_asr = whisper.load_model(MODEL_SIZE, device=device)
        print("Whisper ready")
    return _model_asr


def transcribe(audio_path, device="cuda"):
    """
    Transcribe an audio file to text.

    Args:
        audio_path: path to the audio file (.mp3, .wav, etc.)
        device:     "cuda" for GPU (recommended), "cpu" as fallback

    Returns:
        transcript (str): the transcribed text, stripped of whitespace
    """
    model  = get_model(device=device)
    result = model.transcribe(
        audio_path,
        language       = LANGUAGE,
        initial_prompt = INITIAL_PROMPT,
        fp16           = (device == "cuda"),
    )
    return result["text"].strip()
