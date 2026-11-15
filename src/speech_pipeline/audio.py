"""Audio loading. Every stage of the pipeline works on 16 kHz mono float32 audio."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000


def load_audio(path: str | Path) -> np.ndarray:
    """Load a wav file as a 1-D float32 array in [-1, 1] at 16 kHz.

    A wrong sample rate is an error rather than a silent resample, so a badly
    prepared file is caught before it reaches the models. Multi-channel audio
    is averaged to mono."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != SAMPLE_RATE:
        raise ValueError(
            f"{Path(path).name}: sample rate is {sr} Hz, expected {SAMPLE_RATE}. Convert with:\n"
            f"  ffmpeg -i <input> -ac 1 -ar {SAMPLE_RATE} -c:a pcm_s16le <output>.wav"
        )
    return audio.mean(axis=1)


def duration(audio: np.ndarray) -> float:
    """Length of a 16 kHz signal in seconds."""
    return len(audio) / SAMPLE_RATE