"""Download the model files the pipeline needs into models/.
Files that are already there are skipped, so it is safe to run again."""
import urllib.request

from speech_pipeline.config import load_config, project_path, relative

SHERPA = "https://github.com/k2-fsa/sherpa-onnx/releases/download"

# file name in models/ -> download URL
MODELS = {
    "silero_vad.onnx": f"{SHERPA}/asr-models/silero_vad.onnx",
}


def download(url: str, dest) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"ok    {relative(dest)}")
        return
    print(f"get   {relative(dest)} ...", end=" ", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")   # no half-written files if interrupted
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    print(f"{dest.stat().st_size / 1e6:.1f} MB")


def main() -> None:
    folder = project_path(load_config(), "models")
    folder.mkdir(exist_ok=True)
    for name, url in MODELS.items():
        download(url, folder / name)
    print("done")


if __name__ == "__main__":
    main()