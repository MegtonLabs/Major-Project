import cv2
import numpy as np
import easyocr
import os
import sys

# Force UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')

class SmartChequeLoader:
    def __init__(self, use_gpu=True):
        # Define local model path
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_dir = os.path.join(project_root, "models", "easyocr")
        os.makedirs(self.model_dir, exist_ok=True)

        print(f"[Data] Initializing Smart Loader (Storage: {self.model_dir})...")
        
        # Initialize EasyOCR with custom model path
        self.reader = easyocr.Reader(
            ['en'], 
            gpu=use_gpu, 
            model_storage_directory=self.model_dir,
            download_enabled=True
        )
        
        # Keywords that should appear upright
        self.ANCHORS = ["IFSC", "BANK", "PAY", "RUPEES", "A/C", "NO", "VALID", "ONLY", "SIGNATURE"]

    def load_and_correct(self, image_path):
        """
        Loads an image, detects its orientation, corrects it, 
        and returns the fixed image + DETAILED OCR results.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        original_img = cv2.imread(image_path)
        if original_img is None:
            raise ValueError(f"Could not read image: {image_path}")

        best_img = original_img
        best_angle = 0
        best_score = -1

        print(f"[Info] Analyzing orientation for: {os.path.basename(image_path)}")

        # Phase 1: Quick Scan for Orientation (Fast Mode)
        for angle in [0, 90, 180, 270]:
            # Rotate image
            if angle == 0:
                img_rot = original_img
            elif angle == 90:
                img_rot = cv2.rotate(original_img, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                img_rot = cv2.rotate(original_img, cv2.ROTATE_180)
            elif angle == 270:
                img_rot = cv2.rotate(original_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Optimization: Resize for faster text detection (width=800)
            h, w = img_rot.shape[:2]
            scale = 800 / w if w > 800 else 1.0
            small_img = cv2.resize(img_rot, (int(w*scale), int(h*scale)))

            # Quick detect (detail=0 gives just text list)
            try:
                text_results = self.reader.readtext(small_img, detail=0)
                text_blob = " ".join(text_results).upper()
            except Exception:
                text_blob = ""

            # Score this orientation
            score = 0
            for anchor in self.ANCHORS:
                if anchor in text_blob:
                    score += 1
            
            # Critical anchors give higher confidence
            if "IFSC" in text_blob: score += 3
            if "BANK" in text_blob: score += 2

            if score > best_score:
                best_score = score
                best_angle = angle
                best_img = img_rot

        print(f"[OK] Detected Orientation: {best_angle}° (Confidence Score: {best_score})")
        
        # Phase 2: Full OCR on the Winner (Detailed Mode)
        print("   -> Running full detail OCR on corrected image...")
        full_results = self.reader.readtext(best_img) 
        
        return best_img, best_angle, full_results
