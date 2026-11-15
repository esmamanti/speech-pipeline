"""Speaker diarization: who speaks when.

Two stages:

1. Clustering (pyannote segmentation-3.0 + speaker embedding + clustering, run on CPU
   through sherpa-onnx; no PyTorch, no Hugging Face token):
     - segmentation finds speaker changes and overlapping speech in 10 s windows,
     - an embedding model turns each local speaker into a "voice fingerprint",
     - clustering groups the fingerprints; each group becomes one speaker.

2. Small-speaker merging (the role of pyannote's `min_cluster_size`):
   clustering leaves extra "speakers" with only a few seconds of talk, mostly from
   overlapping speech where the fingerprint mixes two voices. Each such fragment is
   given to the real speaker whose average voice (centroid) it resembles most.
   Fingerprints are taken from overlap-free speech only.

The number of speakers is never given to the system; it is estimated.
Speaker labels (SPK_00, SPK_01, ...) follow the order in which people first speak.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sherpa_onnx

from speech_pipeline.audio import SAMPLE_RATE
from speech_pipeline.config import project_path
from speech_pipeline.types import SpeakerTurn

MIN_CLEAN_AUDIO = 1.0   # s of overlap-free speech needed for a reliable fingerprint
MIN_EMBED_AUDIO = 0.5   # s; shorter excerpts are widened around their centre

Interval = tuple[float, float]


@dataclass
class Reassignment:
    """One merging decision, kept so it can be printed and checked."""
    talk_time: float          # seconds of speech in the fragment
    target: str | None        # speaker it was merged into; None = kept separate
    similarity: float         # cosine similarity to that speaker's centroid


# ================================================================ public API
def diarize(audio: np.ndarray, cfg: dict) -> list[SpeakerTurn]:
    """Speaker turns for 16 kHz mono audio."""
    turns, _ = diarize_with_report(audio, cfg)
    return turns


def diarize_with_report(audio: np.ndarray, cfg: dict) -> tuple[list[SpeakerTurn], list[Reassignment]]:
    """Speaker turns plus the small-speaker merging decisions."""
    params = cfg["diarization"]

    turns = _cluster(audio, cfg)
    report: list[Reassignment] = []
    if params["merge_small_speakers"]:
        turns, report = _merge_small_speakers(audio, turns, cfg)
    turns = _join_adjacent(turns, max_gap=params["min_duration_off"])

    turns, names = _relabel_by_first_appearance(turns)
    report = [Reassignment(r.talk_time, names.get(r.target), r.similarity) for r in report]
    return turns, report


def talk_time(turns: list[SpeakerTurn]) -> dict[str, float]:
    """Total seconds per speaker."""
    totals: dict[str, float] = {}
    for turn in turns:
        totals[turn.speaker] = totals.get(turn.speaker, 0.0) + turn.duration
    return dict(sorted(totals.items()))


# ================================================================ stage 1: clustering
def _cluster(audio: np.ndarray, cfg: dict) -> list[SpeakerTurn]:
    params = cfg["diarization"]
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=_model_path(cfg, params["segmentation"])),
            num_threads=params["num_threads"],
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=_model_path(cfg, params["embedding"]),
            num_threads=params["num_threads"],
        ),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=params["num_speakers"] or -1,   # -1 = let the threshold decide
            threshold=params["cluster_threshold"],
        ),
        min_duration_on=params["min_duration_on"],
        min_duration_off=params["min_duration_off"],
    )
    if not config.validate():
        raise RuntimeError("Invalid diarization config")

    pipeline = sherpa_onnx.OfflineSpeakerDiarization(config)
    result = pipeline.process(audio).sort_by_start_time()
    return [SpeakerTurn(r.start, r.end, f"SPK_{r.speaker:02d}") for r in result]


# ================================================================ stage 2: small-speaker merging
def _merge_small_speakers(audio: np.ndarray, turns: list[SpeakerTurn],
                          cfg: dict) -> tuple[list[SpeakerTurn], list[Reassignment]]:
    params = cfg["diarization"]
    times = talk_time(turns)
    limit = max(params["min_speaker_time"], params["min_speaker_ratio"] * sum(times.values()))
    small = sorted((s for s, t in times.items() if t < limit), key=lambda s: -times[s])
    large = [s for s, t in times.items() if t >= limit]
    if not small or not large:
        return turns, []

    extractor = _load_extractor(cfg)
    centroids = _speaker_centroids(audio, turns, large, extractor)

    new_label: dict[str, str] = {}
    report: list[Reassignment] = []
    for fragment in small:
        vector = _fragment_fingerprint(audio, turns, fragment, extractor)
        candidates = _candidate_centroids(turns, fragment, centroids)
        best = max(candidates, key=lambda s: float(vector @ candidates[s]))
        similarity = float(vector @ candidates[best])

        # Merge when the voice is similar enough. Fragments under MIN_CLEAN_AUDIO hold too
        # little voice to judge; keeping them would always create a fake speaker.
        if similarity >= params["min_similarity"] or times[fragment] < MIN_CLEAN_AUDIO:
            new_label[fragment] = best
            report.append(Reassignment(times[fragment], best, similarity))
        else:
            report.append(Reassignment(times[fragment], None, similarity))

    merged = [SpeakerTurn(t.start, t.end, new_label.get(t.speaker, t.speaker)) for t in turns]
    return merged, report


def _speaker_centroids(audio, turns, speakers, extractor) -> dict[str, np.ndarray]:
    """Average voice of each speaker: duration-weighted mean of overlap-free fingerprints."""
    centroids = {}
    for speaker in speakers:
        vectors, weights = [], []
        for turn in (t for t in turns if t.speaker == speaker):
            pieces = _without_overlap(turn, turns)
            if _total(pieces) >= MIN_CLEAN_AUDIO:
                vectors.append(_embed(extractor, audio, pieces))
                weights.append(turn.duration)
        if vectors:
            centroids[speaker] = _unit(np.average(vectors, axis=0, weights=weights))
    return centroids


def _fragment_fingerprint(audio, turns, fragment, extractor) -> np.ndarray:
    """Fingerprint of a small speaker, from overlap-free speech if there is enough of it."""
    own = [t for t in turns if t.speaker == fragment]
    pieces = [p for t in own for p in _without_overlap(t, turns)]
    if _total(pieces) < MIN_CLEAN_AUDIO:
        pieces = [(t.start, t.end) for t in own]   # fully overlapped: use what there is
    return _embed(extractor, audio, pieces)


def _candidate_centroids(turns, fragment, centroids) -> dict[str, np.ndarray]:
    """Speakers the fragment may be merged into. A fragment lying inside someone else's
    turn is an interruption by another person, so that speaker is excluded."""
    own = [t for t in turns if t.speaker == fragment]
    interrupted = {o.speaker for t in own for o in turns
                   if o.speaker != fragment and o.start <= t.start and o.end >= t.end}
    candidates = {s: c for s, c in centroids.items() if s not in interrupted}
    return candidates or centroids


# ================================================================ audio helpers
def _model_path(cfg: dict, name: str) -> str:
    path = project_path(cfg, "models") / name
    if not path.exists():
        raise FileNotFoundError(f"Diarization model missing: models/{name}. Run: python scripts/download_models.py")
    return str(path)


def _load_extractor(cfg: dict) -> sherpa_onnx.SpeakerEmbeddingExtractor:
    params = cfg["diarization"]
    return sherpa_onnx.SpeakerEmbeddingExtractor(sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=_model_path(cfg, params["embedding"]), num_threads=params["num_threads"]))


def _embed(extractor, audio: np.ndarray, pieces: list[Interval]) -> np.ndarray:
    """One unit-length fingerprint for the joined audio of `pieces`."""
    total = len(audio) / SAMPLE_RATE
    chunks = []
    for start, end in pieces:
        if end - start < MIN_EMBED_AUDIO:   # widen very short excerpts around their centre
            centre = (start + end) / 2
            start = max(0.0, centre - MIN_EMBED_AUDIO / 2)
            end = min(total, centre + MIN_EMBED_AUDIO / 2)
        chunks.append(audio[int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)])

    stream = extractor.create_stream()
    stream.accept_waveform(sample_rate=SAMPLE_RATE, waveform=np.concatenate(chunks))
    stream.input_finished()
    return _unit(np.array(extractor.compute(stream)))


def _without_overlap(turn: SpeakerTurn, turns: list[SpeakerTurn]) -> list[Interval]:
    """Parts of `turn` where nobody else is talking at the same time."""
    pieces: list[Interval] = [(turn.start, turn.end)]
    for other in turns:
        if other.speaker != turn.speaker:
            pieces = [p for piece in pieces for p in _subtract(piece, (other.start, other.end))]
    return [p for p in pieces if p[1] - p[0] > 0.05]


def _subtract(piece: Interval, cut: Interval) -> list[Interval]:
    """`piece` minus `cut` (0, 1 or 2 intervals)."""
    (start, end), (cut_start, cut_end) = piece, cut
    if cut_end <= start or cut_start >= end:
        return [piece]
    left = [(start, cut_start)] if cut_start > start else []
    right = [(cut_end, end)] if cut_end < end else []
    return left + right


def _total(pieces: list[Interval]) -> float:
    return sum(end - start for start, end in pieces)


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


# ================================================================ tidying up
def _join_adjacent(turns: list[SpeakerTurn], max_gap: float) -> list[SpeakerTurn]:
    """Join back-to-back turns of the same speaker separated by at most `max_gap` seconds
    (merging can leave one person's speech split around a reassigned fragment)."""
    joined: list[SpeakerTurn] = []
    for turn in sorted(turns, key=lambda t: t.start):
        last = joined[-1] if joined else None
        if last and last.speaker == turn.speaker and turn.start - last.end <= max_gap:
            last.end = max(last.end, turn.end)
        else:
            joined.append(SpeakerTurn(turn.start, turn.end, turn.speaker))
    return joined


def _relabel_by_first_appearance(turns: list[SpeakerTurn]) -> tuple[list[SpeakerTurn], dict[str, str]]:
    """Rename speakers so the first person to talk is SPK_00, the second SPK_01, ...
    (clustering numbers are arbitrary). Returns the new turns and the old->new mapping."""
    names: dict[str, str] = {}
    for turn in sorted(turns, key=lambda t: t.start):
        names.setdefault(turn.speaker, f"SPK_{len(names):02d}")
    return [SpeakerTurn(t.start, t.end, names[t.speaker]) for t in turns], names


# ================================================================ command line
def main() -> None:
    """python -m speech_pipeline.diarization <wav> [--threshold 0.75] [--no-merge] [--turns 20]"""
    import argparse
    import time

    from speech_pipeline.audio import duration, load_audio
    from speech_pipeline.config import load_config, relative

    parser = argparse.ArgumentParser(description="Run speaker diarization on one wav file.")
    parser.add_argument("wav")
    parser.add_argument("--threshold", type=float, help="override cluster_threshold")
    parser.add_argument("--no-merge", action="store_true", help="skip small-speaker merging")
    parser.add_argument("--turns", type=int, default=0, help="print the first N speaker turns")
    args = parser.parse_args()

    cfg = load_config()
    if args.threshold is not None:
        cfg["diarization"]["cluster_threshold"] = args.threshold
    if args.no_merge:
        cfg["diarization"]["merge_small_speakers"] = False

    audio = load_audio(args.wav)
    start = time.perf_counter()
    turns, report = diarize_with_report(audio, cfg)
    elapsed = time.perf_counter() - start
    length = duration(audio)

    params = cfg["diarization"]
    print(f"{relative(args.wav)}  (threshold {params['cluster_threshold']}, "
          f"merge {'on' if params['merge_small_speakers'] else 'off'})")
    print(f"  speakers found : {len(talk_time(turns))}")
    print(f"  speaker turns  : {len(turns)}")
    print(f"  processing     : {elapsed:.1f} s for {length:.0f} s audio (RTF {elapsed / length:.3f})")
    print("  talk time:")
    for speaker, seconds in talk_time(turns).items():
        print(f"    {speaker}: {seconds:6.1f} s")
    if report:
        print("  small speakers found by clustering:")
        for r in report:
            action = f"merged into {r.target}" if r.target else "kept as a separate speaker"
            print(f"    {r.talk_time:5.1f} s -> {action} (similarity {r.similarity:.2f})")
    for t in turns[:args.turns]:
        print(f"    {t.start:7.2f} - {t.end:7.2f}  {t.speaker}")


if __name__ == "__main__":
    main()