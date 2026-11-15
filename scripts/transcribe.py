"""Speaker-attributed transcription of one or more wav files (16 kHz mono).

Usage:
  python scripts/transcribe.py data/processed/debate_clip.wav
  python scripts/transcribe.py data/processed/*.wav --formats json txt

Output, for every input file, in outputs/ (or --out-dir):
  <name>.json  <name>.txt  <name>.srt  <name>.rttm
"""
import argparse
import glob
from pathlib import Path

from speech_pipeline.audio import load_audio
from speech_pipeline.config import load_config, project_path, relative
from speech_pipeline.diarization import talk_time
from speech_pipeline.export import FORMATS, export
from speech_pipeline.pipeline import Pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("wavs", nargs="+", help="wav files (wildcards allowed)")
    parser.add_argument("--out-dir", type=Path, help="output folder (default: outputs/)")
    parser.add_argument("--formats", nargs="+", choices=FORMATS, default=list(FORMATS))
    args = parser.parse_args()

    # PowerShell does not expand *.wav itself, so expand wildcards here
    files = [f for pattern in args.wavs for f in (sorted(glob.glob(pattern)) or [pattern])]

    cfg = load_config()
    out_dir = args.out_dir or project_path(cfg, "outputs")

    print("loading models ...", flush=True)
    with Pipeline(cfg) as pipeline:
        for path in files:
            print(f"\n{relative(path)}", flush=True)
            result = pipeline.run(load_audio(path))

            t = result.timings
            print(f"  {result.duration:.0f} s audio in {t['total']:.1f} s (RTF {t['total'] / result.duration:.3f}) | "
                  f"VAD {t['vad']:.1f} s, diarization {t['diarization']:.1f} s, ASR {t['asr']:.1f} s "
                  f"(in parallel), alignment {t['alignment']:.2f} s")
            print(f"  language {result.language} | {len(talk_time(result.turns))} speakers | "
                  f"{len(result.utterances)} utterances | {len(result.words)} words")
            for written in export(result, out_dir, Path(path).stem, args.formats):
                print(f"  saved {relative(written)}")


if __name__ == "__main__":
    main()