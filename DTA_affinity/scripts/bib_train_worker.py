#!/usr/bin/env python3
"""Run the unchanged trainer and record wall time and peak CUDA allocation."""

import argparse
import json
import runpy
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-output", type=Path, required=True)
    args, training_args = parser.parse_known_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is disabled")
    if args.runtime_output.exists():
        raise RuntimeError(f"Runtime record already exists: {args.runtime_output}")
    torch.cuda.reset_peak_memory_stats()
    started = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    trainer = Path(__file__).with_name("train_nexusdti_regressor.py")
    sys.argv = [str(trainer), *training_args]
    runpy.run_path(str(trainer), run_name="__main__")
    torch.cuda.synchronize()
    runtime = {"started_utc": started, "completed_utc": datetime.now(timezone.utc).isoformat(),
               "wall_seconds": time.perf_counter() - start,
               "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
               "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
               "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
               "scope": "Whole affinity trainer including setup, evaluation and output; excludes pretrained feature extraction"}
    args.runtime_output.parent.mkdir(parents=True, exist_ok=True)
    args.runtime_output.write_text(json.dumps(runtime, indent=2) + "\n")
    print(json.dumps({"runtime": runtime}), flush=True)


if __name__ == "__main__":
    main()
