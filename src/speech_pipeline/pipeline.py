"""The whole pipeline: audio -> speaker-attributed transcript.

                 ┌──► diarization (CPU) ──► who spoke when ──┐
audio ──► VAD ───┤                                          ├──► alignment ──► utterances
                 └──► ASR (GPU) ──────────► words + times ──┘

Diarization and ASR do not depend on each other, so they run at the same time:
diarization in a separate process, ASR in this one. A separate process is needed
because sherpa-onnx keeps Python's global lock (GIL) while it works; in a thread it
would block the ASR thread and nothing would really run in parallel.
The Whisper model is loaded once, in the constructor, and reused for every file.
"""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from speech_pipeline.alignment import align
from speech_pipeline.asr import Transcriber
from speech_pipeline.audio import duration
from speech_pipeline.diarization import diarize
from speech_pipeline.types import SpeakerTurn, SpeechRegion, Utterance, Word
from speech_pipeline.vad import detect_speech


@dataclass
class PipelineResult:
    duration: float                       # audio length (s)
    language: str
    regions: list[SpeechRegion]
    turns: list[SpeakerTurn]
    words: list[Word]
    utterances: list[Utterance]
    timings: dict[str, float] = field(default_factory=dict)   # seconds per stage


class Pipeline:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.transcriber = Transcriber(cfg)                  # slow: load the ASR model only once
        self.diarization_worker = ProcessPoolExecutor(max_workers=1)

    def close(self) -> None:
        """Stop the diarization worker process."""
        self.diarization_worker.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def run(self, audio: np.ndarray) -> PipelineResult:
        timings: dict[str, float] = {}
        total_start = time.perf_counter()

        regions, timings["vad"] = _timed(detect_speech, audio, self.cfg)

        # Start diarization in the worker process, run ASR here meanwhile, then collect both
        diarization_job = self.diarization_worker.submit(_timed, diarize, audio, self.cfg)
        (words, language), timings["asr"] = _timed(self.transcriber.transcribe, audio, regions)
        turns, timings["diarization"] = diarization_job.result()

        utterances, timings["alignment"] = _timed(align, words, turns, self.cfg)
        timings["total"] = time.perf_counter() - total_start

        return PipelineResult(duration(audio), language, regions, turns, words, utterances, timings)


def _timed(function, *args):
    """Run `function(*args)` and return (result, seconds it took)."""
    start = time.perf_counter()
    result = function(*args)
    return result, time.perf_counter() - start