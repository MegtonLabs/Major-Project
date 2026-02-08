import easyocr
import cv2
import os
import random
import re
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
DATA_DIR = r"D:\Major-Project\data\cheques"
OUTPUT_DIR = r"D:\Major-Project\experiments\vision_debug"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Regex Rules (The "Logic" Supervisor) ---
REGEX_RULES = {
    "IFSC": r"^[A-Z]{4}0[A-Z0-9]{6}$",  # 4 letters, 0, 6 alphanum
    "MICR": r"^[0-9]{9}$",              # 9 digits
    "ACCOUNT": r"^[0-9]{9,18}$",        # 9-18 digits
    "AMOUNT": r"^[0-9,]+\.\d{2}$"       # Number with decimal
}

def load_random_image():
    files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not files:
        print("❌ No images found in", DATA_DIR)
        return None, None
    
    filename = random.choice(files)
    path = os.path.join(DATA_DIR, filename)
    print(f"📂 Loading: {filename}")
    return cv2.imread(path), filename

def visualize_detections(img, results, filename):
    plt.figure(figsize=(15, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # Draw boxes
    ax = plt.gca()
    
    print("\n🔍 OCR Results & Logic Check:")
    print("-" * 40)
    
    found_ifsc = False
    
    for (bbox, text, prob) in results:
        # EasyOCR returns bbox as [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
        (tl, tr, br, bl) = bbox
        x = int(tl[0])
        y = int(tl[1])
        w = int(tr[0] - tl[0])
        h = int(bl[1] - tl[1])
        
        # Clean text
        text = text.strip().upper().replace(" ", "")
        
        # --- The "Zero-Shot" Logic Check ---
        label = "TEXT"
        color = 'yellow'
        linewidth = 1
        
        # Check against Regex Rules
        if re.match(REGEX_RULES["IFSC"], text):
            label = "✅ IFSC"
            color = 'lime' # Green for success
            linewidth = 3
            found_ifsc = True
            print(f"   >>> FOUND IFSC: {text}")
            
        elif re.match(REGEX_RULES["MICR"], text):
            label = "✅ MICR"
            color = 'cyan'
            linewidth = 2
            
        elif re.match(REGEX_RULES["ACCOUNT"], text):
            label = "❓ A/C?"
            color = 'orange'
            linewidth = 2

        # Draw on plot
        rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=linewidth)
        ax.add_patch(rect)
        ax.text(x, y - 5, f"{label}: {text}", fontsize=8, color='white', 
                bbox=dict(facecolor=color, alpha=0.7))

    plt.axis('off')
    save_path = os.path.join(OUTPUT_DIR, f"debug_{filename}")
    plt.savefig(save_path)
    print(f"\n💾 Saved visualization to: {save_path}")
    
    if found_ifsc:
        print("\n✅ SUCCESS: Unsupervised Logic successfully identified the IFSC code!")
    else:
        print("\n⚠️ NOTE: IFSC not found by strict regex. RL agent will learn to fix this.")

def main():
    print("🚀 Starting Vision System Test...")
    
    # Initialize Reader (English)
    # gpu=True will use your RTX 3060 automatically
    reader = easyocr.Reader(['en'], gpu=True) 
    
    img, filename = load_random_image()
    if img is None: return

    # Run Inference
    print("🧠 Running OCR (on GPU)...")
    results = reader.readtext(img)
    
    # Visualize
    visualize_detections(img, results, filename)

if __name__ == "__main__":
    main()
