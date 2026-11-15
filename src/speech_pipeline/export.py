"""Write pipeline results to files.

  .json  everything, for programs (the voice agent's next stage reads this)
  .txt   readable transcript, one utterance per line
  .srt   subtitles, to watch the result on the video
  .rttm  speaker turns in the standard diarization format, for evaluation tools
"""
from __future__ import annotations

import json
from pathlib import Path

from speech_pipeline.alignment import format_time
from speech_pipeline.diarization import talk_time
from speech_pipeline.pipeline import PipelineResult

FORMATS = ("json", "txt", "srt", "rttm")


def export(result: PipelineResult, out_dir: Path, name: str, formats=FORMATS) -> list[Path]:
    """Write the chosen formats as <out_dir>/<name>.<format>; returns the written paths."""
    writers = {"json": _json, "txt": _txt, "srt": _srt, "rttm": _rttm}
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = out_dir / f"{name}.{fmt}"
        path.write_text(writers[fmt](result, name), encoding="utf-8")
        paths.append(path)
    return paths


def _json(result: PipelineResult, name: str) -> str:
    data = {
        "file": name,
        "duration": round(result.duration, 3),
        "language": result.language,
        "speakers": {s: round(t, 2) for s, t in talk_time(result.turns).items()},
        "timings": {k: round(v, 2) for k, v in result.timings.items()},
        "utterances": [
            {
                "speaker": u.speaker,
                "start": round(u.start, 3),
                "end": round(u.end, 3),
                "text": u.text,
                "words": [{"text": w.text, "start": round(w.start, 3), "end": round(w.end, 3),
                           "probability": round(w.probability, 3)} for w in u.words],
            }
            for u in result.utterances
        ],
    }
    return json.dumps(data, ensure_ascii=False, indent=2)


def _txt(result: PipelineResult, name: str) -> str:
    lines = [f"[{format_time(u.start)} - {format_time(u.end)}] {u.speaker}: {u.text}" for u in result.utterances]
    return "\n".join(lines) + "\n"


def _srt(result: PipelineResult, name: str) -> str:
    blocks = []
    for i, u in enumerate(result.utterances, start=1):
        blocks.append(f"{i}\n{_srt_time(u.start)} --> {_srt_time(u.end)}\n{u.speaker}: {u.text}\n")
    return "\n".join(blocks)


def _rttm(result: PipelineResult, name: str) -> str:
    return "".join(f"SPEAKER {name} 1 {t.start:.3f} {t.duration:.3f} <NA> <NA> {t.speaker} <NA> <NA>\n"
                   for t in sorted(result.turns, key=lambda t: t.start))


def _srt_time(seconds: float) -> str:
    """75.3 -> '00:01:15,300'"""
    ms = round(seconds * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"