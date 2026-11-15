"""Data structures shared by all pipeline stages.

    VAD          -> list[SpeechRegion]   where is speech?
    Diarization  -> list[SpeakerTurn]    who speaks when?
    ASR          -> list[Word]           what is said, and when?
    Alignment    -> list[Utterance]      who said what?

All times are in seconds from the start of the audio.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SpeechRegion:
    """A stretch of audio that contains speech (speaker unknown)."""
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class SpeakerTurn:
    """A stretch of audio where one speaker talks."""
    start: float
    end: float
    speaker: str

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Word:
    """One recognised word with its timing."""
    text: str
    start: float
    end: float
    probability: float = 1.0   # ASR confidence, 0..1


@dataclass
class Utterance:
    """A sentence or phrase by one speaker: the final output of the pipeline."""
    speaker: str
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)