"""Voice activity detection: find where speech is, ignoring silence and noise.

Silero VAD (via sherpa-onnx) scores every 32 ms frame with a speech probability.
Frames above the threshold are speech; neighbouring speech frames are joined into
regions. Regions are then padded slightly so word onsets/endings are not cut.
"""
from __future__ import annotations

import numpy as np
import sherpa_onnx

from speech_pipeline.audio import SAMPLE_RATE
from speech_pipeline.config import project_path
from speech_pipeline.types import SpeechRegion

FRAME = 512  # samples per VAD step = 32 ms at 16 kHz


def detect_speech(audio: np.ndarray, cfg: dict) -> list[SpeechRegion]:
    """Return the speech regions of 16 kHz mono audio."""
    params = cfg["vad"]
    model = project_path(cfg, "models") / params["model"]
    if not model.exists():
        raise FileNotFoundError(f"VAD model missing: models/{model.name}. Run: python scripts/download_models.py")

    vad_cfg = sherpa_onnx.VadModelConfig()
    vad_cfg.silero_vad.model = str(model)
    vad_cfg.silero_vad.threshold = params["speech_threshold"]
    vad_cfg.silero_vad.min_speech_duration = params["min_speech_duration"]
    vad_cfg.silero_vad.min_silence_duration = params["min_silence_duration"]
    vad_cfg.silero_vad.max_speech_duration = params["max_speech_duration"]
    vad_cfg.silero_vad.window_size = FRAME
    vad_cfg.sample_rate = SAMPLE_RATE

    total = len(audio) / SAMPLE_RATE
    vad = sherpa_onnx.VoiceActivityDetector(vad_cfg, buffer_size_in_seconds=total + 5)

    regions: list[SpeechRegion] = []

    def collect() -> None:
        # Move every finished speech segment out of the detector's queue
        while not vad.empty():
            start = vad.front.start / SAMPLE_RATE
            regions.append(SpeechRegion(start, start + len(vad.front.samples) / SAMPLE_RATE))
            vad.pop()

    # Feed the audio frame by frame, the same way a live microphone stream would arrive
    for i in range(0, len(audio), FRAME):
        vad.accept_waveform(audio[i:i + FRAME])
        collect()
    vad.flush()   # close a region that is still open at the end of the audio
    collect()

    return pad_and_merge(regions, params["padding"], total)


def pad_and_merge(regions: list[SpeechRegion], padding: float, total: float) -> list[SpeechRegion]:
    """Extend each region by `padding` seconds on both sides (within the audio)
    and merge regions that now touch or overlap."""
    merged: list[SpeechRegion] = []
    for region in sorted(regions, key=lambda r: r.start):
        start = max(0.0, region.start - padding)
        end = min(total, region.end + padding)
        if merged and start <= merged[-1].end:
            merged[-1].end = max(merged[-1].end, end)
        else:
            merged.append(SpeechRegion(start, end))
    return merged


if __name__ == "__main__":
    # Quick look at the VAD output:  python -m speech_pipeline.vad data/processed/debate_clip.wav
    import argparse

    from speech_pipeline.audio import duration, load_audio
    from speech_pipeline.config import load_config, relative

    parser = argparse.ArgumentParser(description="Run VAD on one wav file and print the speech regions.")
    parser.add_argument("wav")
    args = parser.parse_args()

    audio = load_audio(args.wav)
    regions = detect_speech(audio, load_config())

    total = duration(audio)
    speech = sum(r.duration for r in regions)
    gaps = [b.start - a.end for a, b in zip(regions, regions[1:])]
    print(f"{relative(args.wav)}")
    print(f"  regions        : {len(regions)}")
    print(f"  speech         : {speech:.1f} s of {total:.1f} s ({100 * speech / total:.0f} %)")
    print(f"  longest region : {max(r.duration for r in regions):.1f} s")
    if gaps:
        print(f"  longest pause  : {max(gaps):.1f} s")
    print("  first regions:")
    for r in regions[:10]:
        print(f"    {r.start:7.2f} - {r.end:7.2f}  ({r.duration:5.2f} s)")
    if len(regions) > 10:
        print(f"    ... {len(regions) - 10} more")