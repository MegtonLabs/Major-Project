"""
Cheque Verification System — Demo
===================================
Demonstrates the full pipeline (detection → verification → extraction)
on one or more sample cheque images with rich terminal output.

Usage:
    python demo.py                          # auto-picks first image in test_data/
    python demo.py --image path/to/img.jpg  # specific image
    python demo.py --all                    # all images in test_data/
    python demo.py --extract-only img.jpg   # skip verification, only extract fields
"""

import sys
import time
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

TEST_DATA_DIR = BASE_DIR / "test_data"


# ── Terminal helpers ──────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
VIOLET = "\033[95m"
DIM    = "\033[2m"


def _banner(title: str):
    width = 60
    print(f"\n{BOLD}{CYAN}{'═' * width}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'═' * width}{RESET}")


def _section(title: str):
    print(f"\n{BOLD}{VIOLET}── {title} {'─' * (54 - len(title))}{RESET}")


def _ok(msg):  print(f"  {GREEN}✓{RESET}  {msg}")
def _warn(msg): print(f"  {YELLOW}⚠{RESET}  {msg}")
def _err(msg):  print(f"  {RED}✗{RESET}  {msg}")
def _info(msg): print(f"  {DIM}→{RESET}  {msg}")


def _verdict_color(verdict: str) -> str:
    if verdict == "GENUINE": return GREEN
    if verdict == "FORGED":  return RED
    return YELLOW


# ── Demo runner ───────────────────────────────────────────────────────────────

def run_demo(image_path: str, extract_only: bool = False):
    _banner(f"Cheque Verification Demo — {Path(image_path).name}")

    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    _info(f"Image: {image_path}  ({w}×{h}px)")

    if not extract_only:
        # ── Phase 1: Detection ────────────────────────────────────────────────
        _section("Phase 1 — Signature Detection (Falcon Perception)")
        t0 = time.time()
        bbox = None
        detect_method = "heuristic"
        try:
            from agent_studio import _load_falcon, _detect
            _info("Loading Falcon Perception 0.6B …")
            _load_falcon()
            dets = _detect(img, "signature", task="segmentation")
            if dets:
                det = max(dets, key=lambda d: (
                    (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1])
                    if "bbox" in d else 0
                ))
                if "bbox" in det:
                    bbox = det["bbox"]
                    detect_method = "falcon-perception"
                    _ok(f"Signature detected via {detect_method}  bbox={bbox}  ({round(time.time()-t0,2)}s)")
            else:
                _warn("Falcon found no detections — using heuristic fallback")
        except Exception as e:
            _err(f"Falcon failed: {e}")

        if bbox is None:
            bbox = [int(w * 0.50), int(h * 0.58), w, h]
            _warn(f"Heuristic bbox={bbox}")

        # ── Phase 1b: Line Sweep ──────────────────────────────────────────────
        _section("Phase 1b — Line Sweep Tight Crop")
        try:
            from detection.Line_Sweep.lineSweepDetect import line_sweep_with_bounds
            crop = img.crop(bbox)
            ls_result = line_sweep_with_bounds(crop)
            if ls_result.get("success") and ls_result["image"].height >= 30:
                sig_img = ls_result["image"]
                _ok(f"Line Sweep cropped  {sig_img.size}")
            else:
                sig_img = crop
                _warn("Line Sweep did not improve crop — using Falcon bbox crop")
        except Exception as e:
            sig_img = img.crop(bbox)
            _err(f"Line Sweep failed: {e}")

        # ── Phase 2: Verification ─────────────────────────────────────────────
        _section("Phase 2 — Signature Verification (Signature SVM)")
        t_ver = time.time()
        verdict = "UNKNOWN"
        confidence = 0.0
        verifier_name = "none"

        try:
            from signature_svm.verifier import verify_signature_pil as svm_verify, is_trained as svm_ready
            if svm_ready():
                label, conf = svm_verify(sig_img)
                verdict       = "GENUINE" if label == "REAL" else "FORGED" if label == "FORGED" else "INCONCLUSIVE"
                confidence    = conf
                verifier_name = "Signature SVM"
        except Exception:
            pass

        vc = _verdict_color(verdict)
        dt_ver = round(time.time() - t_ver, 2)
        print(f"\n  {BOLD}Verdict: {vc}{verdict}{RESET}  "
              f"(confidence={confidence:.1%}, model={verifier_name}, {dt_ver}s)")

    # ── Phase 3: Field Extraction ─────────────────────────────────────────────
    _section("Phase 3 — Field Extraction (Gemma 4 E2B)")
    t_ext = time.time()
    try:
        from detection.ocr_extractor import extract_cheque_fields
        _info("Loading Gemma 4 E2B …")
        fields = extract_cheque_fields(img)
        dt_ext = fields.pop("_duration_s", round(time.time() - t_ext, 2))

        if "error" in fields:
            _err(f"Extraction error: {fields['error']}")
        else:
            _ok(f"Extracted in {dt_ext}s")
            print()
            field_order = [
                ("account_holder",  "Account Holder"),
                ("bank_name",       "Bank Name"),
                ("branch_name",     "Branch"),
                ("cheque_number",   "Cheque No."),
                ("date",            "Date"),
                ("payee_name",      "Payee"),
                ("amount_numeric",  "Amount (₹)"),
                ("amount_words",    "Amount (words)"),
                ("signature_present","Signature"),
                ("ifsc_code",       "IFSC Code"),
                ("account_number",  "Account No."),
            ]
            for key, label in field_order:
                val = fields.get(key)
                if val is not None and val != "":
                    print(f"  {CYAN}{label:<20}{RESET} {val}")
                else:
                    print(f"  {DIM}{label:<20} —{RESET}")
    except Exception as e:
        _err(f"Extraction failed: {e}")

    print(f"\n{DIM}Done.{RESET}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def find_test_images() -> list:
    """Return image paths from test_data/."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    images = [p for p in TEST_DATA_DIR.iterdir()
              if p.suffix.lower() in exts] if TEST_DATA_DIR.exists() else []
    return sorted(images)


def main():
    p = argparse.ArgumentParser(description="Cheque Verification Demo")
    p.add_argument("--image",        metavar="PATH", help="Path to a cheque image")
    p.add_argument("--all",          action="store_true", help="Process all images in test_data/")
    p.add_argument("--extract-only", action="store_true", help="Skip verification, only extract fields")
    args = p.parse_args()

    if args.image:
        images = [args.image]
    elif args.all:
        images = find_test_images()
        if not images:
            print(f"[WARN] No images found in {TEST_DATA_DIR}")
            return
    else:
        images = find_test_images()
        if not images:
            print("[WARN] No images in test_data/. Use --image <path>.")
            return
        images = [images[0]]

    for img_path in images:
        run_demo(str(img_path), extract_only=args.extract_only)


if __name__ == "__main__":
    main()
