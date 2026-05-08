"""
Local eSignify Signature Verifier
=================================

This module implements the local SVM-style verifier from the eSignify project:

1. Preprocess signature images to binary ink crops.
2. Extract SIFT descriptors.
3. Build a Bag-of-Visual-Words vocabulary.
4. Add the 12 geometric/contour features used by eSignify.
5. Train a local LinearSVC once per Python process.
6. Predict the uploaded/cropped signature locally.

No HTTP API, external server, TensorFlow, or remote model call is used here.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.cluster.vq import kmeans, vq
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

_BASE = Path(__file__).resolve().parent
if str(_BASE) not in sys.path:
    sys.path.insert(0, str(_BASE))

import features
import preproc

DATA_GENUINE = _BASE / "data" / "genuine"
DATA_FORGED = _BASE / "data" / "forged"
VOCAB_SIZE = 500
MIN_CONFIDENCE = 0.65

_MODEL_CACHE: dict | None = None


class ForgeryServerUnavailable(RuntimeError):
    """Kept for compatibility with existing app code."""


def _image_files(folder: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)


def is_ready() -> bool:
    """Return True when the local eSignify training folders are available."""
    return (
        DATA_GENUINE.exists()
        and DATA_FORGED.exists()
        and len(_image_files(DATA_GENUINE)) > 0
        and len(_image_files(DATA_FORGED)) > 0
    )


def is_trained() -> bool:
    """Alias used by the existing app/status code."""
    return is_ready()


def _sift_detector():
    try:
        return cv2.xfeatures2d.SIFT_create()
    except AttributeError:
        return cv2.SIFT_create()


def _preprocess_path(path: Path) -> np.ndarray:
    """Use the local eSignify preprocessing implementation for dataset images."""
    return preproc.preproc(str(path), display=False)


def _preprocess_pil(img: Image.Image) -> np.ndarray:
    """Preprocess an uploaded PIL crop using the same binary-crop idea."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    grey = np.mean(arr, axis=2)
    bin_img = preproc.greybin(grey)
    rows, cols = np.where(bin_img == 1)
    if rows.size == 0 or cols.size == 0:
        raise ValueError("No signature ink was found in the crop.")
    sign_img = bin_img[rows.min(): rows.max() + 1, cols.min(): cols.max() + 1]
    return (255 * sign_img).astype("uint8")


def _sift_descriptors(binary_img: np.ndarray) -> np.ndarray | None:
    detector = _sift_detector()
    _kp, des = detector.detectAndCompute(binary_img, None)
    if des is None or len(des) == 0:
        return None
    return des.astype("float32")


def _safe_div(a: float, b: float) -> float:
    if b == 0 or not math.isfinite(float(b)):
        return 0.0
    value = float(a) / float(b)
    return value if math.isfinite(value) else 0.0


def _geometric_features(binary_img: np.ndarray) -> np.ndarray:
    """Return the 12 geometric features described by eSignify."""
    try:
        aspect_ratio, bounding_area, hull_area, contour_area = features.get_contour_features(
            binary_img.copy(), display=False
        )
        ratio = features.Ratio(binary_img.copy())
        centroid_0, centroid_1 = features.Centroid(binary_img.copy())
        eccentricity, solidity = features.EccentricitySolidity(binary_img.copy())
        (skew_0, skew_1), (kurt_0, kurt_1) = features.SkewKurtosis(binary_img.copy())
        values = [
            aspect_ratio,
            _safe_div(hull_area, bounding_area),
            _safe_div(contour_area, bounding_area),
            ratio,
            centroid_0,
            centroid_1,
            eccentricity,
            solidity,
            skew_0,
            skew_1,
            kurt_0,
            kurt_1,
        ]
    except Exception:
        values = [0.0] * 12

    arr = np.asarray(values, dtype=np.float32)
    arr[~np.isfinite(arr)] = 0.0
    return arr


def _histogram_from_descriptors(des: np.ndarray | None, vocabulary: np.ndarray, size: int) -> np.ndarray:
    hist = np.zeros(size, dtype=np.float32)
    if des is None or len(des) == 0 or len(vocabulary) == 0:
        return hist
    words, _dist = vq(des, vocabulary)
    for word in words:
        if 0 <= word < size:
            hist[word] += 1.0
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def _sample_from_path(path: Path) -> dict:
    binary = _preprocess_path(path)
    return {
        "path": path,
        "binary": binary,
        "des": _sift_descriptors(binary),
        "geom": _geometric_features(binary),
    }


def _build_model() -> dict:
    if not is_ready():
        raise ForgeryServerUnavailable(
            "Local eSignify dataset is missing. Expected images in "
            f"{DATA_GENUINE} and {DATA_FORGED}."
        )

    samples = []
    labels = []
    descriptor_blocks = []

    for path in _image_files(DATA_GENUINE):
        try:
            sample = _sample_from_path(path)
        except Exception:
            continue
        samples.append(sample)
        labels.append(1)  # REAL
        if sample["des"] is not None:
            descriptor_blocks.append(sample["des"])

    for path in _image_files(DATA_FORGED):
        try:
            sample = _sample_from_path(path)
        except Exception:
            continue
        samples.append(sample)
        labels.append(0)  # FORGED
        if sample["des"] is not None:
            descriptor_blocks.append(sample["des"])

    if len(samples) < 4 or not descriptor_blocks:
        raise ForgeryServerUnavailable("Not enough usable signature samples to train the local SVM.")

    descriptors = np.vstack(descriptor_blocks).astype("float32")
    vocab_size = min(VOCAB_SIZE, len(descriptors))
    vocabulary, _variance = kmeans(descriptors, vocab_size, 1)

    rows = []
    for sample in samples:
        bow = _histogram_from_descriptors(sample["des"], vocabulary, VOCAB_SIZE)
        rows.append(np.concatenate([bow, sample["geom"]]).astype("float32"))

    x = np.vstack(rows)
    y = np.asarray(labels, dtype=np.int32)

    scaler = StandardScaler().fit(x)
    x_scaled = scaler.transform(x)
    classifier = LinearSVC(class_weight="balanced", max_iter=10000, random_state=42)
    classifier.fit(x_scaled, y)

    return {
        "classifier": classifier,
        "scaler": scaler,
        "vocabulary": vocabulary,
    }


def _model() -> dict:
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        _MODEL_CACHE = _build_model()
    return _MODEL_CACHE


def verify(pil_img: Image.Image) -> tuple[str, float]:
    """Verify a cropped signature image locally.

    Returns:
        ("REAL", confidence), ("FORGED", confidence), or
        ("UNKNOWN", confidence) when the SVM margin is too weak.
    """
    model = _model()
    binary = _preprocess_pil(pil_img)
    des = _sift_descriptors(binary)
    if des is None or len(des) == 0:
        return "UNKNOWN", 0.50

    geom = _geometric_features(binary)
    bow = _histogram_from_descriptors(des, model["vocabulary"], VOCAB_SIZE)
    x = np.concatenate([bow, geom]).astype("float32").reshape(1, -1)
    x = model["scaler"].transform(x)

    pred = int(model["classifier"].predict(x)[0])
    label = "REAL" if pred == 1 else "FORGED"

    margin = float(model["classifier"].decision_function(x)[0])
    if not math.isfinite(margin):
        confidence = 0.5
    else:
        confidence = 1.0 / (1.0 + math.exp(-min(abs(margin), 50.0)))
    confidence = max(0.50, min(0.99, confidence))
    if confidence < MIN_CONFIDENCE:
        return "UNKNOWN", confidence
    return label, confidence


def verify_signature_pil(pil_img: Image.Image) -> tuple[str, float]:
    """Compatibility alias used by agent.py and any local UI code."""
    return verify(pil_img)


def verify_signature_file(path: str | Path) -> tuple[str, float]:
    """Verify a local cropped signature image file."""
    img = Image.open(path).convert("RGB")
    return verify(img)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a cropped signature locally with eSignify SVM.")
    parser.add_argument("image", help="Path to a cropped signature image")
    args = parser.parse_args()

    label, confidence = verify_signature_file(args.image)
    if label == "REAL":
        verdict = "GENUINE"
    elif label == "FORGED":
        verdict = "FORGED"
    else:
        verdict = "INCONCLUSIVE"
    print(f"{verdict} confidence={confidence:.3f}")


if __name__ == "__main__":
    main()
