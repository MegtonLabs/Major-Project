import easyocr
import cv2
import os
import re
import numpy as np
from tqdm import tqdm

# Configuration
INPUT_DIR = r"D:\Major-Project\data\cheques"
OUTPUT_DIR = r"D:\Major-Project\data\cheques_corrected"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Anchor words commonly found on Indian cheques (uppercase)
ANCHORS = ["IFSC", "BANK", "PAY", "RUPEES", "A/C", "ACCOUNT", "VALID", "ONLY", "SIGNATURE"]

class ChequePreprocessor:
    def __init__(self):
        # Initialize EasyOCR (using GPU if available)
        print("🚀 Initializing OCR Engine...")
        self.reader = easyocr.Reader(['en'], gpu=True)

    def detect_orientation(self, img):
        """
        Tries 0, 90, 180, 270 degree rotations.
        Returns the best rotated image and the angle found.
        """
        best_score = -1
        best_img = img
        best_angle = 0

        # Try 4 orientations
        for angle in [0, 90, 180, 270]:
            if angle == 0:
                rotated = img
            elif angle == 90:
                rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                rotated = cv2.rotate(img, cv2.ROTATE_180)
            elif angle == 270:
                rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Fast OCR on resized image for speed (width=800)
            h, w = rotated.shape[:2]
            scale = 800 / w if w > 800 else 1.0
            small = cv2.resize(rotated, (int(w*scale), int(h*scale)))
            
            # Get text (detail=0 for just list of strings)
            try:
                results = self.reader.readtext(small, detail=0)
                text_blob = " ".join(results).upper()
                
                # Score based on anchors
                score = 0
                for anchor in ANCHORS:
                    if anchor in text_blob:
                        score += 1
                
                # Bonus for finding "IFSC" specifically as it's critical
                if "IFSC" in text_blob:
                    score += 2
                
                if score > best_score:
                    best_score = score
                    best_img = rotated
                    best_angle = angle
                    
            except Exception as e:
                print(f"Error during OCR at {angle}°: {e}")

        return best_img, best_angle, best_score

    def process_all(self):
        files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"📂 Found {len(files)} images to process.")

        for f in tqdm(files, desc="Correcting Orientation"):
            path = os.path.join(INPUT_DIR, f)
            img = cv2.imread(path)
            
            if img is None:
                continue

            corrected_img, angle, score = self.detect_orientation(img)
            
            # Save
            save_path = os.path.join(OUTPUT_DIR, f)
            cv2.imwrite(save_path, corrected_img)
            
            if angle != 0:
                tqdm.write(f"🔄 Fixed {f}: Rotated {angle}° (Score: {score})")

if __name__ == "__main__":
    processor = ChequePreprocessor()
    processor.process_all()
    print(f"\n✅ Done! Corrected images saved to: {OUTPUT_DIR}")
