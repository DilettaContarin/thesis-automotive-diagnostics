"""
tts.py
------
Text-to-Speech output for the online pipeline.

Current implementation uses espeak — a rule-based speech synthesiser
available natively on Linux (including Google Colab).

Privacy note:
  espeak operates entirely offline with zero external dependencies,
  consistent with the privacy-preserving design of this system.
  Cloud-based alternatives (e.g. gTTS) were rejected because they
  transmit query/response data to external servers.

Production deployment:
  Replace espeak with pyttsx3 (OS-native, offline) or Coqui TTS
  (neural, significantly better voice quality, still offline) for
  more natural speech output while preserving the local design.

Installation (Linux / Google Colab):
  apt-get install -y espeak espeak-data libespeak1
"""

import subprocess
import os


def speak(text, speed=150, save_path=None):
    """
    Convert text to speech using espeak and save as WAV.

    Args:
        text:       the answer text to synthesise
        speed:      words per minute (default 150; lower = slower/clearer)
        save_path:  optional path to save the WAV file (defaults to /tmp)

    Returns:
        path (str): path to the generated WAV file
    """
    path = save_path or "/tmp/answer_audio.wav"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    subprocess.run(
        ["espeak", "-v", "en", "-s", str(speed), "-w", path, text],
        check  = True,
        stderr = subprocess.DEVNULL,
    )
    return path


def speak_and_play(text, speed=150, save_path=None):
    """
    Synthesise text and play it inline in a Jupyter/Colab environment.
    Falls back to just saving if IPython is not available.
    """
    path = speak(text, speed=speed, save_path=save_path)

    try:
        from IPython.display import Audio, display
        display(Audio(path, autoplay=True))
    except ImportError:
        print(f"[TTS] Audio saved to {path} (IPython not available for playback)")

    return path
