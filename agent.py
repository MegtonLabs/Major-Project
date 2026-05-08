"""
Cheque Verification Agent
==========================
Programmatic API for the full cheque verification pipeline.
Can be used standalone or imported by other scripts.

Pipeline (mirrors cheque_studio.py SSE events):
  1. Falcon Perception 0.6B  — signature region detection (segmentation)
  2. Line Sweep              — tight signature crop
  3. Signature SVM           — GENUINE / FORGED classification
  4. Gemma 4 E2B (mlx_vlm)  — structured cheque field extraction (11 fields)

Example:
    from agent import ChequeVerificationAgent
    agent = ChequeVerificationAgent()
    result = agent.run("path/to/cheque.jpg")
    print(result)
"""

import sys
import time
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))


class ChequeVerificationAgent:
    """Full cheque verification pipeline agent."""

    def __init__(self):
        self._falcon_loaded = False
        self._gemma_loaded  = False

    # ── Model loading ─────────────────────────────────────────────────────────

    def _ensure_falcon(self):
        if self._falcon_loaded:
            return
        from agent_studio import _load_falcon
        _load_falcon()
        self._falcon_loaded = True

    def _ensure_gemma(self):
        if self._gemma_loaded:
            return
        from agent_studio import _load_gemma
        _load_gemma()
        self._gemma_loaded = True

    # ── Phase 1: Detection ────────────────────────────────────────────────────

    def detect_signature(self, img: Image.Image) -> dict:
        """
        Locate the signature region using Falcon Perception.
        Falls back to heuristic bottom-right crop if Falcon finds nothing.
        Returns: {"bbox": [x1,y1,x2,y2], "method": str, "duration_s": float}
        """
        from agent_studio import _detect, _render_detections

        t0 = time.time()
        bbox = None
        method = "heuristic"
        annotated = img.copy()

        try:
            self._ensure_falcon()
            dets = _detect(img, "signature", task="segmentation")
            if dets:
                det = max(dets, key=lambda d: (
                    (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1])
                    if "bbox" in d else 0
                ))
                if "bbox" in det:
                    bbox = det["bbox"]
                    method = "falcon-perception"
                    annotated = _render_detections(img, [det], "signature")
        except Exception as e:
            print(f"[Agent] Falcon detection failed ({e}), using heuristic.")

        if bbox is None:
            w, h = img.size
            bbox = [int(w * 0.50), int(h * 0.58), w, h]
            method = "heuristic"

        return {
            "bbox":       bbox,
            "method":     method,
            "annotated":  annotated,
            "duration_s": round(time.time() - t0, 2),
        }

    # ── Phase 1b: Line Sweep ──────────────────────────────────────────────────

    def line_sweep_crop(self, img: Image.Image, bbox: list) -> dict:
        """
        Apply Line Sweep algorithm to tightly crop the signature.
        Returns: {"image": PIL, "bounds": dict, "success": bool}
        """
        from detection.Line_Sweep.lineSweepDetect import line_sweep_with_bounds

        crop = img.crop(bbox)
        result = line_sweep_with_bounds(crop)
        if result.get("success") and result.get("image") is not None:
            sig = result["image"]
            if sig.height >= 30:
                return result
        return {"image": crop, "bounds": {}, "success": False}

    # ── Phase 2: Verification ─────────────────────────────────────────────────

    def verify_signature(self, sig_img: Image.Image) -> dict:
        """
        Classify the cropped signature as GENUINE, FORGED, or INCONCLUSIVE.
        Uses Signature SVM (SIFT BoVW + LinearSVC, model.pkl) — no TensorFlow.
        Returns: {"verdict": str, "confidence": float, "model": str, "duration_s": float}
        """
        t0 = time.time()

        from signature_svm.verifier import verify_signature_pil, is_trained

        if not is_trained():
            return {
                "verdict":    "UNKNOWN",
                "confidence": 0.0,
                "model":      "none",
                "error":      "signature_svm/model.pkl not found",
                "duration_s": round(time.time() - t0, 2),
            }

        label, conf = verify_signature_pil(sig_img)
        if label == "REAL":
            verdict = "GENUINE"
        elif label == "FORGED":
            verdict = "FORGED"
        else:
            verdict = "INCONCLUSIVE"
        model_name = "Signature SVM"

        return {
            "verdict":    verdict,
            "confidence": conf,
            "model":      model_name,
            "note":       "Low SVM margin; enrolment/reference signatures are needed for a reliable identity decision." if verdict == "INCONCLUSIVE" else "",
            "duration_s": round(time.time() - t0, 2),
        }

    # ── Phase 3: Field Extraction ─────────────────────────────────────────────

    def extract_fields(self, img: Image.Image) -> dict:
        """
        Extract 11 structured fields from the cheque using Gemma 4 E2B.
        Returns: dict with keys matching CHEQUE_FIELDS in ocr_extractor.py
        """
        from detection.ocr_extractor import extract_cheque_fields
        self._ensure_gemma()
        return extract_cheque_fields(img)

    # ── Full pipeline ─────────────────────────────────────────────────────────

    def run(self, image_path: str) -> dict:
        """
        Run the complete cheque verification pipeline on a local image file.
        Returns a result dict with all phases combined.
        """
        t_total = time.time()
        img = Image.open(image_path).convert("RGB")

        result = {
            "image_path": image_path,
            "detection":  {},
            "line_sweep": {},
            "verification": {},
            "extraction": {},
            "total_duration_s": 0.0,
        }

        # Phase 1 — Detection
        det = self.detect_signature(img)
        result["detection"] = {
            "bbox":       det["bbox"],
            "method":     det["method"],
            "duration_s": det["duration_s"],
        }

        # Phase 1b — Line Sweep
        ls = self.line_sweep_crop(img, det["bbox"])
        result["line_sweep"] = {"success": ls.get("success", False)}
        sig_img = ls.get("image") or img.crop(det["bbox"])

        # Phase 2 — Verification
        ver = self.verify_signature(sig_img)
        result["verification"] = ver

        # Phase 3 — Field Extraction
        try:
            fields = self.extract_fields(img)
            fields.pop("_duration_s", None)
            result["extraction"] = fields
        except Exception as e:
            result["extraction"] = {"error": str(e)}

        result["total_duration_s"] = round(time.time() - t_total, 2)
        return result


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import argparse

    p = argparse.ArgumentParser(description="Run Cheque Verification Agent")
    p.add_argument("image", help="Path to cheque image")
    p.add_argument("--verify-only", action="store_true", help="Skip field extraction")
    args = p.parse_args()

    agent = ChequeVerificationAgent()
    result = agent.run(args.image)
    print(json.dumps(result, indent=2, ensure_ascii=False))
