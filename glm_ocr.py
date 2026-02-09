#!/usr/bin/env python3
"""
GLM-OCR - Local OCR using Ollama
Usage:
    python glm_ocr.py <image_path>
    python glm_ocr.py <image_path> --mode formula
    python glm_ocr.py <image_path> --mode table
"""
import sys
import argparse
import base64
from pathlib import Path

try:
    import ollama
except ImportError:
    print("Installing ollama package...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ollama"])
    import ollama


def encode_image(image_path: str) -> str:
    """Read and base64 encode an image file."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def run_ocr(image_path: str, mode: str = "text") -> str:
    """
    Run GLM-OCR on an image.
    
    Args:
        image_path: Path to image file
        mode: 'text', 'formula', or 'table'
    
    Returns:
        Extracted text/content
    """
    prompts = {
        "text": "Text Recognition:",
        "formula": "Formula Recognition:",
        "table": "Table Recognition:",
    }
    
    prompt = prompts.get(mode, prompts["text"])
    
    # Resolve path
    path = Path(image_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    
    print(f"Processing: {path}")
    print(f"Mode: {mode}")
    print("-" * 50)
    
    # Call Ollama
    response = ollama.chat(
        model="glm-ocr",
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [str(path)],
            }
        ],
    )
    
    return response["message"]["content"]


def main():
    parser = argparse.ArgumentParser(description="GLM-OCR: Extract text from images")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument(
        "--mode", "-m",
        choices=["text", "formula", "table"],
        default="text",
        help="Recognition mode (default: text)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Save output to file"
    )
    
    args = parser.parse_args()
    
    try:
        result = run_ocr(args.image, args.mode)
        print(result)
        
        if args.output:
            Path(args.output).write_text(result)
            print(f"\nSaved to: {args.output}")
            
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
