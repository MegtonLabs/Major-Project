import os
import cv2
import json
import random
import sys
import shutil

# Force UTF-8 encoding for stdout to handle EasyOCR progress bars
sys.stdout.reconfigure(encoding='utf-8')

# Add project root to path so we can import src
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from src.agents.baseline_agent import BaselineAgent
from src.models.layoutlm_baseline import LayoutLMv3Baseline

# --- CONFIGURATION ---
SANDBOX_DIR = os.path.join(PROJECT_ROOT, "testing_sandbox")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "cheques")

def clear_sandbox():
    """Cleans up the sandbox folder before a new run."""
    if os.path.exists(SANDBOX_DIR):
        shutil.rmtree(SANDBOX_DIR)
    os.makedirs(SANDBOX_DIR, exist_ok=True)
    print(f"[Sandbox] Cleaned folder: {SANDBOX_DIR}")

def run_test(image_path=None):
    # 1. Setup
    clear_sandbox()
    
    # 2. Pick Image
    if not image_path:
        files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not files:
            print("[Error] No images found in data/cheques!")
            return
        image_name = random.choice(files)
        image_path = os.path.join(DATA_DIR, image_name)
    
    print(f"[Sandbox] Testing Image: {image_path}")

    # 3. Initialize Engines
    print("\n[1/3] Initializing Engines...")
    regex_agent = BaselineAgent()
    layout_model = LayoutLMv3Baseline()

    # 4. Run Pipeline
    print("\n[2/3] Processing Image...")
    # Step A: Smart Load & Orient & OCR & Regex
    result = regex_agent.process_cheque(image_path, visualize=False)
    
    # Step B: Get corrected image for LayoutLM
    img, angle, ocr_results = regex_agent.loader.load_and_correct(image_path)
    
    # Step C: Run LayoutLMv3 (Zero-shot)
    logits, words = layout_model.run_inference(image_path, ocr_results)

    # 5. Save Outputs
    print("\n[3/3] Saving Results to Sandbox...")
    
    # Save JSON Report
    report = {
        "input_file": os.path.basename(image_path),
        "orientation_fixed": f"{angle} degrees",
        "extracted_fields": result["fields"],
        "layoutlm_output_shape": str(logits.shape)
    }
    
    json_path = os.path.join(SANDBOX_DIR, "result.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=4)
        
    # Save Visual Debug Image
    # Draw boxes on the corrected image
    debug_img = img.copy()
    for field, data in result["fields"].items():
        bbox = data["bbox"] # [x1, y1, x2, y2]
        cv2.rectangle(debug_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)
        cv2.putText(debug_img, field, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
    img_save_path = os.path.join(SANDBOX_DIR, "debug_output.jpg")
    cv2.imwrite(img_save_path, debug_img)

    print(f"\n[Done] Check the folder: {SANDBOX_DIR}")
    print(f"   -> Report: result.json")
    print(f"   -> Image:  debug_output.jpg")

if __name__ == "__main__":
    # Optional: pass an image path as argument
    target_img = sys.argv[1] if len(sys.argv) > 1 else None
    run_test(target_img)
