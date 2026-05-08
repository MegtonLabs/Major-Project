"""
Compatibility wrapper for the local eSignify SVM verifier.

The original script was evaluation-oriented and did not correctly return the
prediction for the uploaded/test signature image. Keep the old `svm_algo()`
entry point, but route it through `verifier.verify_signature_file()` so existing
project code still works.
"""

from __future__ import annotations

from pathlib import Path

try:
    from .verifier import verify_signature_file
except ImportError:
    from verifier import verify_signature_file


BASE_DIR = Path(__file__).resolve().parent
TEST_DIR = BASE_DIR / "static" / "LineSweep_Results"


def svm_algo():
    """Classify the first signature image in static/LineSweep_Results.

    Returns:
        "Genuine", "Forged", or "Inconclusive" for compatibility with the
        older project code.
    """
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    images = sorted(p for p in TEST_DIR.iterdir() if p.suffix.lower() in exts)
    if not images:
        print("No test images found in folder:", TEST_DIR)
        return "No test images"

    label, confidence = verify_signature_file(images[0])
    if label == "REAL":
        result = "Genuine"
    elif label == "FORGED":
        result = "Forged"
    else:
        result = "Inconclusive"
    print(f"{images[0].name}: {result} confidence={confidence:.3f}")
    return result


if __name__ == "__main__":
    print(svm_algo())
