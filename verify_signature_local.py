"""
Local signature verification entry point.

Usage:
    python verify_signature_local.py path/to/cropped_signature.png

This does not start FastAPI, call an API endpoint, or require an external
server. It uses the local eSignify-style SIFT + geometric-feature SVM verifier
from signature_svm/verifier.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from signature_svm.verifier import verify_signature_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a cropped signature locally.")
    parser.add_argument("image", help="Path to a cropped signature image")
    args = parser.parse_args()

    path = Path(args.image)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    label, confidence = verify_signature_file(path)
    if label == "REAL":
        verdict = "GENUINE"
    elif label == "FORGED":
        verdict = "FORGED"
    else:
        verdict = "INCONCLUSIVE"
    print(f"{verdict} confidence={confidence:.3f}")


if __name__ == "__main__":
    main()
