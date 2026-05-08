"""
Cheque Verify System — FastAPI App
===================================
Entry point for the Cheque Verification Studio.

Usage:
    python app.py
    python app.py --port 8080
    python app.py --host 127.0.0.1 --port 7860

Pipeline:
    Tab 1 — Signature Verification  : Falcon Perception → Line Sweep → Signature SVM
    Tab 2 — Data Extraction         : Gemma 4 E2B (mlx_vlm)
    Tab 3 — Visual Reasoning        : Gemma 4 E2B (mlx_vlm)
"""

import argparse
import sys
from pathlib import Path

# Ensure the app directory is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_args():
    p = argparse.ArgumentParser(description="Cheque Verification System")
    p.add_argument("--host",   default="0.0.0.0",  help="Bind host (default: 0.0.0.0)")
    p.add_argument("--port",   type=int, default=7860, help="Port (default: 7860)")
    p.add_argument("--reload", action="store_true",   help="Enable hot-reload (dev only)")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  Cheque Verification System")
    print("  Falcon Perception 0.6B  |  Gemma 4 E2B  |  Signature SVM")
    print(f"  http://{args.host}:{args.port}")
    print("=" * 60)

    import uvicorn
    from cheque_studio import app
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
