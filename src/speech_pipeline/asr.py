"""Automatic speech recognition: what is said, and when (word by word).

Whisper (large-v3-turbo) through faster-whisper / CTranslate2, on the GPU when one
is available. Only the speech regions found by VAD are transcribed:

  - silence is never sent to Whisper, which prevents its typical "hallucinations"
    (made-up sentences on silent or noisy audio);
  - neighbouring speech regions are grouped into chunks of at most ~30 s, the length
    Whisper was trained on, so it still sees whole sentences with their context.

Every word comes back with its start/end time in the original audio. These timings
are what the alignment step uses to give each word to a speaker.
"""
from __future__ import annotations

import numpy as np

from speech_pipeline.audio import SAMPLE_RATE
from speech_pipeline.config import project_path
from speech_pipeline.gpu import cuda_device_count, enable_cuda_libraries
from speech_pipeline.types import SpeechRegion, Word


class Transcriber:
    """Loads the Whisper model once and transcribes any number of files with it."""

    def __init__(self, cfg: dict):
        params = cfg["asr"]
        self.language = params["language"]          # None = detect from the audio
        self.beam_size = params["beam_size"]
        self.max_chunk = params["max_chunk"]

        model_dir = project_path(cfg, "models") / params["model"]
        if not (model_dir / "model.bin").exists():
            raise FileNotFoundError(f"ASR model missing: models/{params['model']}. "
                                    "Run: python scripts/download_models.py")

        self.device = _choose_device(params["device"])
        self.compute_type = _choose_compute_type(params["compute_type"], self.device)

        from faster_whisper import WhisperModel   # imported here: slow import, needs CUDA DLLs first
        self.model = WhisperModel(str(model_dir), device=self.device, compute_type=self.compute_type)

    def transcribe(self, audio: np.ndarray, regions: list[SpeechRegion]) -> tuple[list[Word], str]:
        """Words (with times in the original audio) and the language that was used."""
        words: list[Word] = []
        language = self.language
        for chunk in group_regions(regions, self.max_chunk):
            excerpt = audio[int(chunk.start * SAMPLE_RATE):int(chunk.end * SAMPLE_RATE)]
            segments, info = self.model.transcribe(
                excerpt,
                language=language,
                beam_size=self.beam_size,
                word_timestamps=True,
                condition_on_previous_text=False,   # avoids repetition loops across chunks
                vad_filter=False,                    # VAD has already been applied
            )
            language = language or info.language    # detect once, then keep it
            for segment in segments:
                for w in segment.words or []:
                    text = w.word.strip()
                    if text:
                        words.append(Word(text, chunk.start + w.start, chunk.start + w.end, w.probability))
        return words, language


def group_regions(regions: list[SpeechRegion], max_length: float) -> list[SpeechRegion]:
    """Join consecutive speech regions into chunks no longer than `max_length` seconds.
    A single region that is already longer stays on its own (Whisper splits it itself)."""
    chunks: list[SpeechRegion] = []
    for region in regions:
        if chunks and region.end - chunks[-1].start <= max_length:
            chunks[-1].end = region.end
        else:
            chunks.append(SpeechRegion(region.start, region.end))
    return chunks


def _choose_device(setting: str) -> str:
    if setting == "auto":
        return "cuda" if cuda_device_count() > 0 else "cpu"
    if setting == "cuda":
        enable_cuda_libraries()
    return setting


def _choose_compute_type(setting: str, device: str) -> str:
    if setting != "auto":
        return setting
    # float16 is fast and accurate on NVIDIA GPUs; int8 keeps CPU runs reasonable
    return "float16" if device == "cuda" else "int8"


# ================================================================ command line
def main() -> None:
    """python -m speech_pipeline.asr <wav> [--lines 15]"""
    import argparse
    import time

    from speech_pipeline.audio import duration, load_audio
    from speech_pipeline.config import load_config, relative
    from speech_pipeline.vad import detect_speech

    parser = argparse.ArgumentParser(description="Transcribe one wav file (VAD + Whisper).")
    parser.add_argument("wav")
    parser.add_argument("--lines", type=int, default=15, help="how many transcript lines to print")
    args = parser.parse_args()

    cfg = load_config()
    audio = load_audio(args.wav)
    length = duration(audio)

    start = time.perf_counter()
    transcriber = Transcriber(cfg)
    load_time = time.perf_counter() - start

    regions = detect_speech(audio, cfg)
    start = time.perf_counter()
    words, language = transcriber.transcribe(audio, regions)
    elapsed = time.perf_counter() - start

    print(f"{relative(args.wav)}")
    print(f"  device         : {transcriber.device} ({transcriber.compute_type})")
    print(f"  model loading  : {load_time:.1f} s")
    print(f"  transcription  : {elapsed:.1f} s for {length:.0f} s audio (RTF {elapsed / length:.3f})")
    print(f"  language       : {language}")
    print(f"  words          : {len(words)}")
    print(f"  low confidence : {sum(w.probability < 0.5 for w in words)} words below 0.5")
    print("  transcript (start of each line in seconds):")
    for i in range(0, min(len(words), args.lines * 12), 12):
        line = words[i:i + 12]
        print(f"    [{line[0].start:6.1f}] {' '.join(w.text for w in line)}")


if __name__ == "__main__":
    main()