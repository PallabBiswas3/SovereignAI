"""Explicit online provisioning only. The application never downloads ASR models."""
from argparse import ArgumentParser
from pathlib import Path


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["tiny.en", "base.en", "tiny", "base"], default="tiny.en")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path(__file__).resolve().parents[1] / "models" / f"faster-whisper-{args.model}"
    from faster_whisper.utils import download_model

    download_model(args.model, output_dir=str(output.resolve()))
    print(f"Local speech model ready at {output.resolve()}")


if __name__ == "__main__":
    main()
