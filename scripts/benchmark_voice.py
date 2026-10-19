"""Measure the local transcription service with a supplied 5-15s recording.

Run from the project root. Never prints the transcript or persists input audio.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--mime", default="audio/wav")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    import psutil
    from app.core.config import get_settings
    from app.multimodal.speech import SpeechService

    settings = get_settings()
    data = args.audio.read_bytes()
    if len(data) > settings.voice_max_upload_bytes:
        parser.error("recording exceeds configured upload size")
    service = SpeechService(settings)
    finished = threading.Event()
    peak_worker_rss = 0

    def sample():
        nonlocal peak_worker_rss
        while not finished.wait(0.02):
            try:
                if service._process is not None:
                    peak_worker_rss = max(peak_worker_rss, psutil.Process(service._process.pid).memory_info().rss)
            except (psutil.Error, ValueError):
                pass

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()

    async def run():
        rows = []
        for index in range(args.runs):
            started = time.perf_counter()
            result = await service.transcribe(data, args.mime)
            result.pop("text")
            result["voice_to_transcript_seconds"] = time.perf_counter() - started
            rows.append({"run": index + 1, **result})
        return rows

    try:
        rows = asyncio.run(run())
        print(json.dumps({
            "device": "cpu", "compute_type": "int8", "cpu_threads": settings.voice_cpu_threads,
            "cpu": __import__("platform").processor(), "logical_cpus": psutil.cpu_count(),
            "peak_sampled_worker_rss_mib": peak_worker_rss / 1024**2,
            "sampling_interval_ms": 20, "runs": rows,
            "scope": "local service: includes worker startup and IPC, excludes microphone capture and HTTP transport",
        }, indent=2))
    finally:
        finished.set()
        sampler.join(timeout=1)
        service.close()


if __name__ == "__main__":
    main()
