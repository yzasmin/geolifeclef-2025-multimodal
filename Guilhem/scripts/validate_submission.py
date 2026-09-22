#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdm.inference import validate_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Kaggle submission format")
    parser.add_argument("--submission", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ok, errors = validate_submission(args.submission)
    if ok:
        print("Submission is valid")
    else:
        print("Submission is invalid")
        for err in errors:
            print(f"- {err}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
