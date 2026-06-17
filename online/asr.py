# asr.py
import whisper

def transcribe(audio_path, model_size="small", language="en"):
    model  = whisper.load_model(model_size, device="cuda")
    result = model.transcribe(
        audio_path,
        language       = language,
        initial_prompt = "Automotive diagnostic report. Car problems, warning lights, engine issues.",
        fp16           = True,
    )
    return result["text"].strip()
