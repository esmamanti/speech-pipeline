# speech-pipeline

First stage of a voice agent: turns an audio recording into a speaker-attributed transcript
("who said what, and when"). Runs locally; no cloud services, no Hugging Face token.

```
                 ┌──► speaker diarization (CPU) ──► who spoke when ──┐
audio ──► VAD ───┤                                                   ├──► alignment ──► transcript
                 └──► ASR (GPU) ──────────────────► words + times ───┘
```

Diarization and ASR are independent and run in parallel (separate processes);
the alignment step gives every recognised word to a speaker.

| Stage | Model / method |
|---|---|
| VAD | Silero VAD (sherpa-onnx) |
| Speaker diarization | pyannote segmentation-3.0 + NeMo TitaNet-S embeddings (sherpa-onnx), plus small-speaker merging |
| ASR | Whisper large-v3-turbo (faster-whisper, CUDA float16) |
| Alignment | word-level timestamps vs. speaker turns, with lone-word smoothing |

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -e .
python scripts/download_models.py      # ~1.7 GB
```

Input audio must be 16 kHz mono wav:
```
ffmpeg -i input.mp4 -ac 1 -ar 16000 -c:a pcm_s16le data/processed/name.wav
```

## Usage

```
python scripts/transcribe.py data/processed/name.wav
```

Writes to `outputs/`:

| File | Content |
|---|---|
| `name.json` | utterances with speaker, times, text and word-level details (input for the next agent stage) |
| `name.txt` | readable transcript |
| `name.srt` | subtitles |
| `name.rttm` | speaker turns (standard diarization format) |

Each stage can also be run on its own:
`python -m speech_pipeline.vad|diarization|asr|alignment <wav>`.
Settings are in `configs/default.yaml`.

## Results on the test clip

Test clip (not in git): 05:32–12:10 of the
[Michigan U.S. Senate final debate](https://youtu.be/Cs6PQe8LqHo) (WXYZ-TV), 398 s, English,
5 speakers (2 candidates, 3 moderators).

| | |
|---|---|
| Speakers found | 5 of 5 (the count is estimated, never given) |
| Words | 1,297 |
| Processing time | 46 s on an RTX 3060 Laptop (RTF 0.11); ASR alone 12 s |

## Known limitations

- **Overlapping speech:** when a moderator interrupts a speaker, a few words around the
  interruption can be given to the wrong person (about 1% of words on the test clip).
- **Diarization threshold:** `cluster_threshold` (0.75) is the middle of the range where the
  result was stable on the test clip (0.7–0.8); it has not been tuned on a labelled dataset.
- **No accuracy metrics yet:** DER and WER need reference labels, which this project does not have.
- **Offline only:** files are processed as a whole; a live voice agent needs a streaming version.

## Project structure

```
configs/default.yaml        all settings
scripts/download_models.py  fetch models into models/
scripts/transcribe.py       wav -> json / txt / srt / rttm
src/speech_pipeline/
    types.py                SpeechRegion, SpeakerTurn, Word, Utterance
    config.py, audio.py     settings and audio loading
    gpu.py                  CUDA libraries on Windows
    vad.py                  where is speech?
    diarization.py          who is speaking?
    asr.py                  what is said?
    alignment.py            who said what?
    pipeline.py             runs all stages
    export.py               writes the output files
```