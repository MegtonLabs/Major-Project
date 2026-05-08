"""
OCR-Based Signature Detection
==============================
Source: Detection Phase / OCR / OCR_Algorithm.py
        (Thapar Institute of Engineering and Technology Capstone Project)

Enhancement: GLM-4V (offline VLM) is used as the primary detector.
             When GLM-4V is unavailable, the original pytesseract keyword-search
             logic ("Please sign above") is used as a faithful fallback.

Original approach:
  1. Load cheque image with OpenCV.
  2. Apply HSV blue-ink mask and Gaussian blur.
  3. Run pytesseract image_to_data() to get per-word bounding boxes.
  4. Find the words "Please" and "above" and derive the signature region
     from their coordinates using fixed scale factors.
  5. Crop and save to OCR_Results/.

This module exposes:
  detect_signature_region(img)  →  {"bbox": [x1,y1,x2,y2], "method": str}
                                   or {"error": str}
  run_ocr_on_folder(input_dir, output_dir)  — batch script mode (original)
"""

import os
import sys
import numpy as np
import cv2
from PIL import Image

# ── Constants ───────────────
_SCALE_Y   = 2
_SCALE_XL  = 2.5
_SCALE_XR  = 0.5

# HSV blue-ink mask (same values as original)
_LOWER_HSV = np.array([103, 79, 60])
_UPPER_HSV = np.array([129, 255, 255])


# ── Primary: GLM-4V offline detection ────────────────────────────────────────

def _glm_locate(img: Image.Image) -> dict:
    """
    Use the locally-loaded GLM-4V model to locate the signature region.
    Returns {"bbox": [x1,y1,x2,y2]} or {} if GLM is unavailable / failed.
    """
    try:
        # Import from the parent detection package
        _det_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _det_dir not in sys.path:
            sys.path.insert(0, _det_dir)
        from detection.ocr_extractor import load_gemma_model, _glm_infer, _parse_json

        if not load_gemma_model():
            return {}

        w, h   = img.size
        prompt = (
            f"Bank cheque image ({w}x{h}px). "
            "Find the authorized signatory / signature field (lower-right area, "
            "labelled 'Please sign above', 'Authorized Signatory' or 'Signature'). "
            "Return ONLY JSON: "
            '{"x1":<left>,"y1":<top>,"x2":<right>,"y2":<bottom>} '
            'or {"error":"not found"} if absent.'
        )
        raw  = _glm_infer(img, prompt, max_new_tokens=128)
        data = _parse_json(raw)

        if "x1" in data and "y1" in data and "x2" in data and "y2" in data:
            bbox = [
                max(0, int(data["x1"])),
                max(0, int(data["y1"])),
                min(w, int(data["x2"])),
                min(h, int(data["y2"])),
            ]
            if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                return {"bbox": bbox}
    except Exception:
        pass
    return {}


# ── Fallback: pytesseract keyword approach ───────────────────────────

def _tesseract_locate(cv_img: np.ndarray, pil_img: Image.Image) -> dict:
    """
    Pytesseract keyword search for signature region.

    Looks for the words 'please' and 'above' in the pytesseract output,
    then applies the original scale-factor formula to derive the signature bbox.
    """
    try:
        import pytesseract
    except ImportError:
        return {"error": "pytesseract not installed"}

    h, w = cv_img.shape[:2]

    # Blue-ink mask (same HSV range + Gaussian blur as original)
    hsv  = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, _LOWER_HSV, _UPPER_HSV)

    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        if cv2.contourArea(c) < 10:
            cv2.drawContours(mask, [c], -1, (0, 0, 0), -1)

    mask = cv2.GaussianBlur(255 - mask, (3, 3), 0)

    data = pytesseract.image_to_data(pil_img)

    please_cd    = [0, 0, 0, 0]
    above_cd     = [0, 0, 0, 0]
    found_please = found_above = False

    for line in data.splitlines():
        d = line.split("\t")
        if len(d) != 12:
            continue
        word = d[11].lower().strip()
        if word == "please":
            please_cd    = [int(d[6]), int(d[7]), int(d[8]), int(d[9])]
            found_please = True
        elif word == "sign":
            pass   # noted but not needed for bbox math
        elif word == "above":
            above_cd   = [int(d[6]), int(d[7]), int(d[8]), int(d[9])]
            found_above = True

    if not (found_please and found_above):
        return {"error": "Keywords 'please'/'above' not found in OCR output"}

    # Scale-factor formula
    length_sign = above_cd[0] + above_cd[3] - please_cd[0]

    x1 = int(please_cd[0] - length_sign * _SCALE_XL)
    y1 = int(please_cd[1] - length_sign * _SCALE_Y)
    x2 = x1 + int((_SCALE_XL + _SCALE_XR + 1) * length_sign)
    y2 = y1 + int(_SCALE_Y * length_sign)

    bbox = [max(0, x1), max(0, y1), min(w, x2), min(h, y2)]
    if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
        return {"bbox": bbox}

    return {"error": "Computed bbox is invalid"}


# ── Public API ────────────────────────────────────────────────────────────────

def detect_signature_region(img: Image.Image) -> dict:
    """
    Detect the signature region on a cheque image.

    Priority chain:
      1. GLM-4V (offline VLM — best accuracy)
      2. pytesseract "Please sign above" keyword search

    Returns:
      {"bbox": [x1,y1,x2,y2], "method": "glm-4v" | "tesseract"}
      or {"error": "<reason>"}
    """
    # 1. GLM-4V
    result = _glm_locate(img)
    if "bbox" in result:
        result["method"] = "glm-4v"
        return result

    # 2. Tesseract fallback
    cv_img = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    result = _tesseract_locate(cv_img, img)
    if "bbox" in result:
        result["method"] = "tesseract"
        return result

    return {"error": result.get("error", "Signature region not found")}


# ── Batch script mode ───────────────────────────

def run_ocr_on_folder(input_dir: str, output_dir: str) -> dict:
    """
    Process all .jpg files in `input_dir`, crop the detected signature region,
    and write the crops to `output_dir`.

    Returns {"processed": int, "total": int}.
    """
    os.makedirs(output_dir, exist_ok=True)
    total = processed = 0

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(".jpg"):
            continue
        total += 1
        path = os.path.join(input_dir, filename)
        img  = Image.open(path).convert("RGB")

        result = detect_signature_region(img)
        if "bbox" not in result:
            print(f"  SKIP {filename}: {result.get('error','no bbox')}")
            continue

        x1, y1, x2, y2 = result["bbox"]
        crop = np.array(img)[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        out_path = os.path.join(output_dir, f"OCR_Result_{filename}")
        cv2.imwrite(out_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        processed += 1
        print(f"  OK  {filename} → {result['method']}")

    print(f"{processed}/{total} files processed successfully.")
    return {"processed": processed, "total": total}


if __name__ == "__main__":
    # Default paths when run directly
    _here       = os.path.dirname(os.path.abspath(__file__))
    _input_dir  = os.path.join(_here, "..", "..", "..", "Our_Dataset", "Testing")
    _output_dir = os.path.join(_here, "OCR_Results")
    run_ocr_on_folder(_input_dir, _output_dir)
