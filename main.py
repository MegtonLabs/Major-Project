"""
Cheque Verify System — Main Entry Point
========================================
Unified launcher with CLI options.

Usage:
    python main.py                        # start web server (default port 7860)
    python main.py --port 8080
    python main.py --demo <image_path>    # run pipeline on a single image
    python main.py --extract <image_path> # extract fields from a cheque image
"""

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))


def cmd_serve(args):
    """Launch the FastAPI / SSE web server."""
    print("=" * 60)
    print("  Cheque Verification System")
    print("  Models:")
    print("    Detection  : Falcon Perception 0.6B (MLX)")
    print("    Extraction : Gemma 4 E2B (mlx_vlm)")
    print("    Reasoning  : Gemma 4 E2B (mlx_vlm)")
    print("    Verifier   : Signature SVM")
    print(f"  Listening at http://{args.host}:{args.port}")
    print("=" * 60)

    import uvicorn
    from cheque_studio import app
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


def cmd_demo(args):
    """Run the full verification pipeline on a local image and print results."""
    from agent import ChequeVerificationAgent
    import json

    image_path = Path(args.demo)
    if not image_path.exists():
        print(f"[ERROR] File not found: {image_path}")
        sys.exit(1)

    agent = ChequeVerificationAgent()
    result = agent.run(str(image_path))
    print(json.dumps(result, indent=2, ensure_ascii=False))


def cmd_extract(args):
    """Extract structured fields from a cheque image."""
    from PIL import Image
    from detection.ocr_extractor import extract_cheque_fields
    import json

    image_path = Path(args.extract)
    if not image_path.exists():
        print(f"[ERROR] File not found: {image_path}")
        sys.exit(1)

    img = Image.open(image_path).convert("RGB")
    print(f"[INFO] Extracting fields from {image_path.name} ...")
    fields = extract_cheque_fields(img)
    print(json.dumps(fields, indent=2, ensure_ascii=False))


def parse_args():
    p = argparse.ArgumentParser(
        description="Cheque Verification System — main launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python main.py                        # start web UI on port 7860
  python main.py --port 8080            # custom port
  python main.py --demo cheque.jpg      # CLI pipeline demo
  python main.py --extract cheque.jpg   # field extraction only
        """,
    )
    p.add_argument("--host",    default="0.0.0.0", help="Bind host")
    p.add_argument("--port",    type=int, default=7860, help="Server port")
    p.add_argument("--reload",  action="store_true", help="Enable hot-reload")
    p.add_argument("--demo",    metavar="IMAGE", help="Run demo pipeline on IMAGE")
    p.add_argument("--extract", metavar="IMAGE", help="Extract cheque fields from IMAGE")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.demo:
        cmd_demo(args)
    elif args.extract:
        cmd_extract(args)
    else:
        cmd_serve(args)
