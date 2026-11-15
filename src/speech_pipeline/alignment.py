"""Alignment: who said what.

Diarization and ASR run independently, so their outputs only meet through time:

    diarization:  SPK_00 [0.0 ---------- 23.4]   SPK_01 [23.6 ----------- 70.2]
    ASR words:    "of"(0.0) "the"(0.2) ... "sir."(23.2)  "Well,"(23.9) ...

Three steps:
  1. assign  - every word goes to the speaker whose turns overlap it the most;
               a word in a gap between turns goes to the nearest turn (if it is close);
  2. smooth  - a lone word whose speaker differs from both neighbours, and which only
               weakly overlaps its own speaker, is boundary jitter: it gets the
               neighbours' speaker;
  3. group   - consecutive words of the same speaker become one utterance; a long pause
               also starts a new one.
"""
from __future__ import annotations

from speech_pipeline.types import SpeakerTurn, Utterance, Word

UNKNOWN = "UNKNOWN"
NO_SPACE_BEFORE = tuple("-'’.,!?;:%)")   # pieces like "-Sayed", "'s", ",000" attach to the previous word


# ================================================================ public API
def align(words: list[Word], turns: list[SpeakerTurn], cfg: dict) -> list[Utterance]:
    """Speaker-attributed utterances from ASR words and diarization turns."""
    params = cfg["alignment"]
    speakers = assign_speakers(words, turns, params["max_distance"])
    speakers = smooth_lone_words(words, speakers, turns, params["min_support"])
    return group_utterances(words, speakers, params["max_pause"])


def format_time(seconds: float) -> str:
    """75.3 -> '01:15.3'"""
    minutes, rest = divmod(seconds, 60)
    return f"{int(minutes):02d}:{rest:04.1f}"


# ================================================================ 1. assign
def assign_speakers(words: list[Word], turns: list[SpeakerTurn], max_distance: float) -> list[str]:
    """One speaker label per word."""
    return [_best_speaker(word, turns, max_distance) for word in words]


def _best_speaker(word: Word, turns: list[SpeakerTurn], max_distance: float) -> str:
    overlaps: dict[str, float] = {}
    for turn in turns:
        shared = _overlap(word.start, word.end, turn.start, turn.end)
        if shared > 0:
            overlaps[turn.speaker] = overlaps.get(turn.speaker, 0.0) + shared
    if overlaps:
        return max(overlaps, key=overlaps.get)

    # The word falls in a gap between turns: take the nearest turn if it is close enough
    nearest = min(turns, key=lambda t: _distance(word, t), default=None)
    if nearest is not None and _distance(word, nearest) <= max_distance:
        return nearest.speaker
    return UNKNOWN


# ================================================================ 2. smooth
def smooth_lone_words(words: list[Word], speakers: list[str], turns: list[SpeakerTurn],
                      min_support: float) -> list[str]:
    """Give a lone word (A A B A A) its neighbours' speaker when the evidence for B is weak:
    less than `min_support` of the word's duration lies inside B's turns."""
    smoothed = list(speakers)
    for i in range(1, len(words) - 1):
        before, current, after = speakers[i - 1], speakers[i], speakers[i + 1]
        if before == after != current and _support(words[i], current, turns) < min_support:
            smoothed[i] = before
    return smoothed


def _support(word: Word, speaker: str, turns: list[SpeakerTurn]) -> float:
    """Share of the word's duration covered by `speaker`'s turns (0..1)."""
    length = max(word.end - word.start, 1e-3)
    covered = sum(_overlap(word.start, word.end, t.start, t.end) for t in turns if t.speaker == speaker)
    return min(covered / length, 1.0)


# ================================================================ 3. group
def group_utterances(words: list[Word], speakers: list[str], max_pause: float) -> list[Utterance]:
    """Consecutive words of one speaker form an utterance; a pause longer than
    `max_pause` seconds starts a new one."""
    utterances: list[Utterance] = []
    for word, speaker in zip(words, speakers):
        last = utterances[-1] if utterances else None
        if last and last.speaker == speaker and word.start - last.end <= max_pause:
            last.words.append(word)
            last.end = word.end
        else:
            utterances.append(Utterance(speaker, word.start, word.end, "", [word]))
    for utterance in utterances:
        utterance.text = join_words(utterance.words)
    return utterances


def join_words(words: list[Word]) -> str:
    """Join Whisper word pieces into readable text ("El" + "-Sayed" -> "El-Sayed", "$64" + ",000")."""
    text = ""
    for word in words:
        piece = word.text
        text += piece if (not text or piece.startswith(NO_SPACE_BEFORE)) else " " + piece
    return text


# ================================================================ small helpers
def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _distance(word: Word, turn: SpeakerTurn) -> float:
    """Time between a word and a turn (0 if they touch or overlap)."""
    return max(turn.start - word.end, word.start - turn.end, 0.0)


# ================================================================ command line
def main() -> None:
    """python -m speech_pipeline.alignment <wav>   (runs VAD, diarization, ASR, then aligns)"""
    import argparse

    from speech_pipeline.asr import Transcriber
    from speech_pipeline.audio import load_audio
    from speech_pipeline.config import load_config, relative
    from speech_pipeline.diarization import diarize
    from speech_pipeline.vad import detect_speech

    parser = argparse.ArgumentParser(description="Speaker-attributed transcript of one wav file.")
    parser.add_argument("wav")
    parser.add_argument("--max", type=int, default=30, help="how many utterances to print")
    args = parser.parse_args()

    cfg = load_config()
    audio = load_audio(args.wav)

    print("VAD ...", flush=True)
    regions = detect_speech(audio, cfg)
    print("diarization ...", flush=True)
    turns = diarize(audio, cfg)
    print("ASR ...", flush=True)
    words, _ = Transcriber(cfg).transcribe(audio, regions)

    params = cfg["alignment"]
    assigned = assign_speakers(words, turns, params["max_distance"])
    smoothed = smooth_lone_words(words, assigned, turns, params["min_support"])
    utterances = group_utterances(words, smoothed, params["max_pause"])

    print(f"\n{relative(args.wav)}")
    print(f"  words          : {len(words)}")
    print(f"  utterances     : {len(utterances)}")
    print(f"  smoothed words : {sum(a != b for a, b in zip(assigned, smoothed))}")
    print(f"  unknown words  : {smoothed.count(UNKNOWN)}")
    print()
    for u in utterances[:args.max]:
        print(f"[{format_time(u.start)} - {format_time(u.end)}] {u.speaker}: {u.text}")
    if len(utterances) > args.max:
        print(f"... {len(utterances) - args.max} more")


if __name__ == "__main__":
    main()