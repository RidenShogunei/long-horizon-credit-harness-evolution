#!/usr/bin/env python3
"""Exit non-zero when free disk under /home falls below the protocol stop line."""

from __future__ import annotations

import argparse
import shutil
import sys


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--path", default="/home/jinxu")
    p.add_argument("--stop-gb", type=float, default=15.0)
    args = p.parse_args()
    free_gb = shutil.disk_usage(args.path).free / (1000**3)
    print(f"free_gb={free_gb:.1f} stop_gb={args.stop_gb}")
    if free_gb < args.stop_gb:
        print("STOP: free disk below protocol budget", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
