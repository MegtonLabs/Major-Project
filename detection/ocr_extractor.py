"""
Cheque Field Extraction — Hybrid VLM pipeline
============================================
Uses Qwen2.5-VL for OCR hint + Gemma 4 for structured extraction.
Designed for Apple Silicon (M-series chips).

Fields extracted (11):
    account_holder, bank_name, branch_name, cheque_number,
    date, payee_name, amount_numeric, amount_words,
    signature_present, ifsc_code, account_number
"""

import re
import json
import time
import tempfile
import os

from PIL import Image, ImageOps, ImageFilter

_qwen_model = None
_qwen_processor = None
QWEN_OCR_MODEL = "mlx-community/Qwen2-VL-2B-4bit"


def _enhance_for_ocr(img: Image.Image) -> Image.Image:
    """Create a high-contrast, upscaled copy for text-heavy cheque reading."""
    enhanced = img.convert("RGB")
    if enhanced.width < 1600:
        scale = 1600 / max(1, enhanced.width)
        enhanced = enhanced.resize(
            (1600, int(enhanced.height * scale)),
            Image.Resampling.LANCZOS,
        )
    enhanced = ImageOps.autocontrast(enhanced)
    enhanced = enhanced.filter(ImageFilter.SHARPEN)
    return enhanced


def _load_qwen_vlm():
    global _qwen_model, _qwen_processor
    if _qwen_model is not None:
        return True
    try:
        from mlx_vlm import load
        print(f"[VLM-OCR] Downloading Qwen2.5-VL model...")
        _qwen_model, _qwen_processor = load(QWEN_OCR_MODEL)
        print(f"[VLM-OCR] Qwen model loaded successfully")
        return True
    except Exception as e:
        print(f"[VLM-OCR] Could not load Qwen2.5-VL: {e}")
        return False


def _run_qwen_vlm_ocr(img: Image.Image) -> str:
    if not _load_qwen_vlm():
        print("[VLM-OCR] Qwen model not loaded, skipping OCR hint")
        return ""
    try:
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config
        config = load_config(QWEN_OCR_MODEL)
        
        ocr_prompt = (
            "You are an OCR system. Extract ALL text from this cheque image. "
            "Return ONLY the raw text found, one item per line. "
            "Include dates, names, numbers, account details, amounts, and all printed text. "
            "Do NOT add explanations. Return ONLY the extracted text."
        )
        
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img.save(tmp.name)
        
        print("[VLM-OCR] Running Qwen2.5-VL for text extraction...")
        fmt = apply_chat_template(_qwen_processor, config, ocr_prompt, num_images=1)
        res = generate(_qwen_model, _qwen_processor, fmt, [tmp.name], verbose=False, max_tokens=1024, temperature=0.1)
        
        os.unlink(tmp.name)
        result = res.text if hasattr(res, "text") else str(res)
        print(f"[VLM-OCR] OCR text extracted, length: {len(result)}")
        return result
    except Exception as e:
        print(f"[VLM-OCR] OCR pass failed: {e}")
        return ""

# ── Field schema ──────────────────────────────────────────────────────────────

CHEQUE_FIELDS = [
    "account_holder",
    "bank_name",
    "branch_name",
    "cheque_number",
    "date",
    "payee_name",
    "amount_numeric",
    "amount_words",
    "signature_present",
    "ifsc_code",
    "account_number",
]

# ── Prompt ────────────────────────────────────────────────────────────────────

_JSON_PROMPT = (
    "You are an expert Indian bank cheque OCR and verification assistant. "
    "Read the entire cheque carefully, including handwritten text, printed bank "
    "details, date boxes, the MICR line, account information, IFSC, amount box, "
    "payee line, and signature area. Return ONLY one valid JSON object with "
    "exactly these keys. Use null only when a field is genuinely not visible:\n"
    '{"account_holder":null,"bank_name":null,"branch_name":null,'
    '"cheque_number":null,"date":null,"payee_name":null,'
    '"amount_numeric":null,"amount_words":null,'
    '"signature_present":null,"ifsc_code":null,"account_number":null}\n\n'
    "Field rules:\n"
    "- account_holder: account owner's printed name, usually top-left/top-center; "
    "do not confuse it with the payee.\n"
    "- bank_name: full bank name or clearest visible bank name.\n"
    "- branch_name: branch name/city/address near the bank name or branch label.\n"
    "- payee_name: handwritten/typed name on the Pay line after Pay/Pay to.\n"
    "- amount_numeric: the NUMBER written in the amount box (right side). "
    "Copy every digit exactly; remove commas/spaces; keep decimals if present. "
    "Examples: '1,50,000' -> '150000'; '25 000' -> '25000'.\n"
    "- amount_words: the amount written in words on the line below 'Pay' "
    "including Rupees/Only when visible.\n"
    "- date: the date as printed in DD/MM/YYYY or DD-MM-YYYY format. "
    "Look for the date box at the top-right of the cheque.\n"
    "- cheque_number: the 6-digit cheque number, often the first numeric block "
    "in the bottom MICR band or labelled Cheque No.\n"
    "- account_number: the account number in the MICR band at the bottom "
    "(typically 9-18 digits).\n"
    "- ifsc_code: 11-character code starting with 4 letters then '0' then 6 alphanumeric.\n"
    "- signature_present: 'yes' if a handwritten signature/stamp-like signature "
    "is visible in the signature box, 'no' if the box is empty.\n"
    "Never return markdown fences, comments, explanations, or extra keys."
)

# ── Lazy inference helpers ─────────────────────────────────────────────────────

_load_gemma_fn = None
_vlm_fn = None


def _ensure_gemma():
    global _load_gemma_fn, _vlm_fn
    if _load_gemma_fn is not None:
        return True
    try:
        import sys
        from pathlib import Path
        _app_dir = str(Path(__file__).resolve().parent.parent)
        if _app_dir not in sys.path:
            sys.path.insert(0, _app_dir)
        from agent_studio import _load_gemma, _vlm
        _load_gemma_fn = _load_gemma
        _vlm_fn = _vlm
        print("[VLM-Extract] Gemma functions loaded from agent_studio")
        return True
    except Exception as e:
        print(f"[VLM-Extract] Could not import from agent_studio: {e}")
        return False


def _glm_infer(img: Image.Image, prompt: str) -> str:
    """Direct inference wrapper for signature localization."""
    if not _ensure_gemma():
        return '{"error": "model not loaded"}'
    try:
        _load_gemma_fn()
    except Exception as e:
        return f'{{"error": "failed to load: {e}"}}'
    return _vlm_fn(img, prompt)


# ── Main API ──────────────────────────────────────────────────────────────────

def load_gemma_model() -> bool:
    """Load Gemma model. Returns True when ready."""
    if not _ensure_gemma():
        return False
    try:
        _load_gemma_fn()
        return True
    except Exception as e:
        print(f"[VLM-Extract] Model load failed: {e}")
        return False


def is_gemma_loaded() -> bool:
    return _load_gemma_fn is not None and _vlm_fn is not None


def extract_cheque_fields(img: Image.Image) -> dict:
    """
    Hybrid cheque field extraction:

      1. Qwen2.5-VL 2B pass     — lightweight VLM OCR on Apple Silicon.
      2. Gemma 4 (mlx_vlm) pass  — fed image + OCR text as a hint, returns JSON.
      3. Regex fallback          — fills any field still null using OCR text.
    """
    print("[VLM-Extract] Starting cheque field extraction...")
    
    if not _ensure_gemma():
        return {
            "error": (
                "Gemma 4 E2B (mlx_vlm) could not be loaded. "
                "Ensure mlx-vlm is installed and agent_studio.py is present."
            )
        }

    try:
        print("[VLM-Extract] Loading Gemma 4 model...")
        _load_gemma_fn()
        print("[VLM-Extract] Gemma model ready")
    except Exception as e:
        return {"error": f"Failed to load Gemma 4 E2B: {e}"}

    t0 = time.time()
    ocr_img = _enhance_for_ocr(img)

    # ── Pass A: Qwen2.5-VL lightweight OCR ────────────────────────────
    print("[VLM-Extract] Step 1: Running Qwen OCR...")
    ocr_text = _run_qwen_vlm_ocr(ocr_img)

    # ── Pass B: Gemma JSON (with OCR text as hint) ────────────────────────
    fields: dict = {}
    try:
        print("[VLM-Extract] Step 2: Running Gemma for structured extraction...")
        prompt = _JSON_PROMPT
        if ocr_text:
            prompt = (
                _JSON_PROMPT
                + "\n\nFor reference, here is the raw text extracted from the "
                "cheque by an OCR engine — use it together with the image to "
                "fill the JSON; trust the image when they disagree:\n\"\"\"\n"
                + ocr_text
                + "\n\"\"\""
            )
        raw = _vlm_fn(ocr_img, prompt)
        fields = _parse_json(raw)
    except Exception as e:
        print(f"[VLM-Extract] Gemma JSON pass failed: {e}")
        fields = {}

    # ── Pass C: regex fallback fills any null/missing field ───────────────
    regex_fields = _parse_raw_to_fields(ocr_text) if ocr_text else {}
    merged = {k: None for k in CHEQUE_FIELDS}
    if isinstance(fields, dict):
        for k in CHEQUE_FIELDS:
            v = fields.get(k)
            if v not in (None, "", "null"):
                merged[k] = v
    for k, v in regex_fields.items():
        if merged.get(k) in (None, "", "null") and v not in (None, "", "null"):
            merged[k] = v

    # ── Pass C2: targeted repair when important fields are still missing ───
    missing = [k for k in CHEQUE_FIELDS if merged.get(k) in (None, "", "null")]
    key_missing = {"bank_name", "date", "payee_name", "amount_numeric", "amount_words"} & set(missing)
    if key_missing and len(missing) >= 3:
        try:
            print(f"[VLM-Extract] Step 3: Filling missing fields: {', '.join(missing)}")
            repair_prompt = (
                "You are repairing an Indian cheque OCR JSON result. "
                "Look again at the image and the OCR text. Return ONLY valid JSON "
                "with the same 11 keys. Keep existing correct values, and fill any "
                "missing fields if visible. Existing JSON:\n"
                + json.dumps(merged, ensure_ascii=False)
                + "\n\nOCR text:\n"
                + (ocr_text or "")
            )
            repair = _parse_json(_vlm_fn(ocr_img, repair_prompt))
            if isinstance(repair, dict):
                for k in CHEQUE_FIELDS:
                    v = repair.get(k)
                    if merged.get(k) in (None, "", "null") and v not in (None, "", "null"):
                        merged[k] = v
        except Exception as e:
            print(f"[VLM-Extract] Missing-field repair failed: {e}")

    # ── Pass D: numeric repair — Gemma sometimes splits long digit runs ───
    merged = _repair_numerics(merged, ocr_text)
    merged = _normalise_fields(merged)

    if isinstance(fields, dict) and "raw_response" in fields and not _looks_valid(fields):
        merged["raw_response"] = fields["raw_response"]
    if ocr_text:
        merged["_ocr_text"] = ocr_text
    merged["_duration_s"] = round(time.time() - t0, 2)
    return merged


# ── Helpers ───────────────────────────────────────────────────────────────────

def _looks_valid(d: dict) -> bool:
    if not isinstance(d, dict):
        return False
    return len(set(CHEQUE_FIELDS) & set(d.keys())) >= 4


def _parse_json(text: str) -> dict:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return {"raw_response": text}


def _clean_text_value(value):
    if value in (None, "", "null"):
        return None
    if isinstance(value, bool):
        return "yes" if value else "no"
    text = re.sub(r"\s+", " ", str(value)).strip(" :-|\t\r\n")
    return text or None


def _normalise_fields(fields: dict) -> dict:
    out = {k: _clean_text_value(fields.get(k)) for k in CHEQUE_FIELDS}
    if out.get("amount_numeric"):
        out["amount_numeric"] = re.sub(r"[^\d.]", "", out["amount_numeric"])
    if out.get("ifsc_code"):
        out["ifsc_code"] = re.sub(r"\s+", "", out["ifsc_code"]).upper()
    if out.get("signature_present"):
        sig = out["signature_present"].lower()
        if sig in {"true", "present", "signed", "available", "visible"}:
            out["signature_present"] = "yes"
        elif sig in {"false", "absent", "unsigned", "not present", "empty"}:
            out["signature_present"] = "no"
    for extra in ("_ocr_text", "_duration_s", "raw_response"):
        if extra in fields:
            out[extra] = fields[extra]
    return out


def _longest_digit_run(text: str, min_len: int = 4) -> str | None:
    """Return the longest contiguous digit sequence in `text`."""
    if not text:
        return None
    runs = re.findall(r"\d+", text)
    runs = [r for r in runs if len(r) >= min_len]
    if not runs:
        return None
    return max(runs, key=len)


def _repair_numerics(fields: dict, ocr_text: str) -> dict:
    """Clean up numeric fields without blindly overwriting correct VLM output.

    Rules:
    - Strip internal whitespace/commas from amount_numeric (but keep the digits).
    - Only replace a field from OCR when the field is genuinely missing/empty —
      never replace a plausible VLM value with an OCR heuristic guess.
    """

    def strip_separators(val: str) -> str:
        """Remove spaces, commas, and other non-digit/non-letter chars from a numeric string."""
        return re.sub(r"[\s,_\-]", "", val)

    out = dict(fields)

    # amount_numeric: strip formatting separators; fallback to regex only if blank
    amt = out.get("amount_numeric")
    if isinstance(amt, str) and amt not in ("", "null"):
        out["amount_numeric"] = strip_separators(amt)
        digits = re.sub(r"\D", "", out["amount_numeric"])
        if len(digits) < 2:
            # Clearly bad — try regex from OCR text
            m = re.search(r'(?:₹|Rs\.?|INR)\s*([\d,]+(?:\.\d{1,2})?)', ocr_text or "", re.IGNORECASE)
            if not m:
                m = re.search(r'\b(\d{3,12}(?:\.\d{1,2})?)\b', ocr_text or "")
            if m:
                out["amount_numeric"] = strip_separators(m.group(1))
    elif not amt:
        m = re.search(r'(?:₹|Rs\.?|INR)\s*([\d,]+(?:\.\d{1,2})?)', ocr_text or "", re.IGNORECASE)
        if m:
            out["amount_numeric"] = strip_separators(m.group(1))

    # account_number: only fill if missing
    if not out.get("account_number") and ocr_text:
        m = re.search(r'(?:A/?c|Account)\s*(?:No\.?|Number)\s*[:\-]?\s*(\d{9,18})', ocr_text, re.IGNORECASE)
        if m:
            out["account_number"] = m.group(1)

    # cheque_number: only fill if missing
    if not out.get("cheque_number") and ocr_text:
        m = re.search(r'(?:Cheque|Ch\.?)?\s*(?:No\.?|Number)\s*[:\-]?\s*(\d{6,10})\b', ocr_text, re.IGNORECASE)
        if m:
            out["cheque_number"] = m.group(1)
        else:
            runs = re.findall(r"\b\d{6}\b", ocr_text)
            if runs:
                out["cheque_number"] = runs[0]

    # ifsc_code: strip internal spaces
    if isinstance(out.get("ifsc_code"), str):
        out["ifsc_code"] = re.sub(r"\s+", "", out["ifsc_code"]).upper()

    return out


def _parse_raw_to_fields(text: str) -> dict:
    """Regex fallback parser for Indian bank cheque layouts."""
    t = re.sub(r"[ \t]+", " ", text.strip())
    lines = [ln.strip(" :-|\t") for ln in t.splitlines() if ln.strip(" :-|\t")]

    def find(pattern, flags=re.IGNORECASE):
        m = re.search(pattern, t, flags)
        return m.group(1).strip() if m else None

    # Date — DD/MM/YYYY, DD-MM-YYYY, D/M/YY, or "15 Apr 2024"
    date = (
        find(r'\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})\b')
        or find(r'\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{2,4})\b')
    )

    # Amount numeric — prefer currency-prefixed; strip commas afterwards
    _amt_raw = (
        find(r'(?:₹|Rs\.?|INR)\s*([\d,]+(?:\.\d{1,2})?)')
        or find(r'(?:Amount|Amt)[\s:]*(?:₹|Rs\.?)?\s*([\d,]+(?:\.\d{1,2})?)')
        or find(r'\b(?:Rupees?|Rs\.?)\s*(\d{2,12}(?:\.\d{1,2})?)\b')
    )
    amount_num = re.sub(r",", "", _amt_raw) if _amt_raw else None

    # Amount in words — "Rupees ... Only" or "... Rupees Only"
    amount_words = (
        find(r'((?:Rupees?|Rs\.?)\s+[^\n\r]{2,120}?\s+[Oo]nly)')
        or find(r'(?:in\s+words?|amount\s+in\s+words?)[\s:]+([^\n\r]{2,120}(?:[Oo]nly)?)')
        or find(r'([A-Z][a-z]+(?:\s+[A-Za-z]+){1,10})\s+[Oo]nly')
    )

    # IFSC — 4 letters + 0 + 6 alphanumeric
    ifsc = find(r'\b([A-Z]{4}0[A-Z0-9]{6})\b')

    digit_runs = re.findall(r"\b\d{6,18}\b", t)

    # Cheque number — 6-digit run labelled or at MICR start
    cheque_no = (
        find(r'(?:Cheque|Ch\.?)\s*(?:No\.?|Number)?\s*[:\-]?\s*(\d{6,10})\b')
        or next((run for run in digit_runs if len(run) == 6), None)
    )

    # Account number — labelled or MICR middle segment
    acct_no = (
        find(r'(?:A/?c\.?|Account)\s*(?:No\.?|Number)?\s*[:\-]?\s*(\d{9,18})\b')
        or next((run for run in digit_runs if 9 <= len(run) <= 18 and run != cheque_no), None)
    )

    # Payee — text after "Pay" up to end-of-line or "or bearer"
    payee = find(
        r'\bPay(?:\s+to)?(?:\s+the\s+order\s+of)?\s+([^\n\r]+?)(?:\s+or\s+bearer|\s+rupees?|\s*$)',
        re.IGNORECASE | re.MULTILINE,
    )
    if payee:
        payee = re.sub(r"\bA/c\s+Payee\b|\bAccount\s+Payee\b", "", payee, flags=re.IGNORECASE).strip(" -")

    # Branch — line after "Branch:" label
    branch = (
        find(r'Branch\s*[:\-]\s*([^\n]+)')
        or find(r'\b([A-Za-z ]+)\s+Branch\b')
    )

    # Account holder — labelled
    account_holder = (
        find(r'(?:A/?c\.?\s*(?:Holder|Name)|Account\s+Holder|Name)\s*[:\-]?\s*([^\n]+)')
        or find(r'For\s+([A-Z][A-Za-z .&]{3,60})')
    )

    # Bank name — first clean line that contains "Bank" or common bank abbreviations
    bank_name = None
    bank_patterns = r'\b(bank|sbi|hdfc|icici|axis|kotak|canara|pnb|union bank|bank of baroda)\b'
    for line in lines[:12]:
        if re.search(bank_patterns, line, re.IGNORECASE) and len(line) > 3:
            bank_name = line
            break

    # Signature present — label found in OCR
    sig_present = None
    if re.search(r'\b(signature|signatory|signed|please sign above)\b', t, re.IGNORECASE):
        sig_present = "yes"

    return {
        "account_holder":    account_holder,
        "bank_name":         bank_name,
        "branch_name":       branch,
        "cheque_number":     cheque_no,
        "date":              date,
        "payee_name":        payee,
        "amount_numeric":    amount_num,
        "amount_words":      amount_words,
        "signature_present": sig_present,
        "ifsc_code":         ifsc,
        "account_number":    acct_no,
    }
