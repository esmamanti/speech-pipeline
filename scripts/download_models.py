"""Download the model files the pipeline needs into models/.
Files that are already there are skipped, so it is safe to run again."""
import tarfile
import urllib.request

from speech_pipeline.config import load_config, project_path, relative

SHERPA = "https://github.com/k2-fsa/sherpa-onnx/releases/download"

# name in models/ (file or folder) -> download URL
SHERPA_MODELS = {
    # VAD
    "silero_vad.onnx": f"{SHERPA}/asr-models/silero_vad.onnx",
    # Diarization: segmentation (archive -> folder) and speaker embedding
    "sherpa-onnx-pyannote-segmentation-3-0":
        f"{SHERPA}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
    "nemo_en_titanet_small.onnx": f"{SHERPA}/speaker-recongition-models/nemo_en_titanet_small.onnx",
}

# ASR: faster-whisper (CTranslate2) model name -> folder in models/
WHISPER_MODELS = {
    "large-v3-turbo": "faster-whisper-large-v3-turbo",
}


def download_file(url: str, dest) -> None:
    if dest.exists():
        print(f"ok    {relative(dest)}")
        return
    is_archive = url.endswith(".tar.bz2")
    target = dest.parent / url.rsplit("/", 1)[-1] if is_archive else dest
    print(f"get   {relative(dest)} ...", end=" ", flush=True)
    tmp = target.with_name(target.name + ".part")   # no half-written files if interrupted
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(target)
    print(f"{target.stat().st_size / 1e6:.1f} MB")
    if is_archive:
        with tarfile.open(target) as tar:
            tar.extractall(dest.parent)   # the archive contains a folder named like `dest`
        target.unlink()
        print(f"      unpacked {relative(dest)}")


def download_whisper(name: str, dest) -> None:
    if (dest / "model.bin").exists():
        print(f"ok    {relative(dest)}")
        return
    from faster_whisper.utils import download_model
    print(f"get   {relative(dest)} (Whisper {name}, ~1.6 GB, this takes a while) ...", flush=True)
    download_model(name, output_dir=str(dest))
    size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
    print(f"      {size / 1e9:.2f} GB")


def main() -> None:
    folder = project_path(load_config(), "models")
    folder.mkdir(exist_ok=True)
    for name, url in SHERPA_MODELS.items():
        download_file(url, folder / name)
    for name, subfolder in WHISPER_MODELS.items():
        download_whisper(name, folder / subfolder)
    print("done")


if __name__ == "__main__":
    main()