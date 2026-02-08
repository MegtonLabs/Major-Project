import cv2
import easyocr
import re
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from src.data.loader import SmartChequeLoader

class BaselineAgent:
    def __init__(self, use_gpu=True):
        print("[Bot] Initializing Baseline Agent...")
        self.loader = SmartChequeLoader(use_gpu=use_gpu)
        self.reader = self.loader.reader 

        # Regex patterns (Position Agnostic - works anywhere)
        self.PATTERNS = {
            "IFSC": r"^[A-Z]{4}0[A-Z0-9]{6}$",
            "MICR": r"^[0-9]{9}$",
            "ACCOUNT": r"^[0-9]{9,18}$",
            "DATE": r"\d{2}[-/]\d{2}[-/]\d{2,4}",
            "AMOUNT": r"^[0-9,]+\.\d{2}$"
        }

    def process_cheque(self, image_path, visualize=False):
        # 1. Load & Correct
        img, angle, results = self.loader.load_and_correct(image_path)
        
        extracted_data = {
            "filename": os.path.basename(image_path),
            "orientation_correction": angle,
            "fields": {}
        }

        # 2. Smart Filter: Only ignore the very bottom edge if it looks like a taskbar date
        # Instead of blind cutting, we just process everything and filter mostly later.
        height, width = img.shape[:2]
        taskbar_threshold = height * 0.96 # Very bottom 4%

        valid_results = []
        for item in results:
            bbox, text, prob = item
            (tl, tr, br, bl) = bbox
            y_center = (tl[1] + bl[1]) / 2
            
            # If text is at the absolute bottom, it MIGHT be taskbar. 
            # But only ignore if it's small text. MICR is usually at bottom but big/clear.
            if y_center > taskbar_threshold:
                continue 
            valid_results.append(item)

        # 3. Regex Extraction (The "Anywhere" Detectors)
        # Pre-calculate MICR zone (Bottom 15%) to avoid confusion
        micr_zone_top = height * 0.85
        
        for (bbox, text, prob) in valid_results:
            clean_text = text.strip().upper().replace(" ", "")
            (tl, tr, br, bl) = bbox
            y_center = (tl[1] + bl[1]) / 2
            
            for field_name, pattern in self.PATTERNS.items():
                if re.match(pattern, clean_text):
                    
                    # --- Special Logic for Account Number ---
                    if field_name == "ACCOUNT":
                        # Rule 1: Ignore if inside MICR zone (Account numbers in MICR are handled by MICR field)
                        if y_center > micr_zone_top:
                            continue
                            
                        # Rule 2: Validation - Is there a label nearby?
                        # Look for "A/c", "No", "SB", "CA" to the LEFT of this number
                        is_valid_acc = False
                        box_left_x = tl[0]
                        
                        # Scan neighbors
                        for (n_bbox, n_text, _) in valid_results:
                            n_right_x = n_bbox[1][0]
                            n_y_center = (n_bbox[0][1] + n_bbox[3][1]) / 2
                            
                            # Neighbor is to the left and on same line
                            if n_right_x < box_left_x and abs(n_y_center - y_center) < 30:
                                if any(k in n_text.upper() for k in ["A/C", "NO", "AC", "SB", "CA", "ACCOUNT"]):
                                    is_valid_acc = True
                                    break
                        
                        # If no label found, enforce stricter constraints (must be long enough)
                        if not is_valid_acc:
                            if len(clean_text) < 11: # Bare numbers < 11 digits are suspicious without label
                                continue
                                
                    # Store Validated Field
                    if field_name in extracted_data["fields"]:
                        # If duplicate, keep the one with a label (priority) or higher confidence
                        current_conf = extracted_data["fields"][field_name]["confidence"]
                        if prob > current_conf:
                             extracted_data["fields"][field_name] = {
                                "text": text,
                                "confidence": float(prob),
                                "bbox": [int(x) for x in bbox[0] + bbox[2]] 
                            }
                    else:
                        extracted_data["fields"][field_name] = {
                            "text": text,
                            "confidence": float(prob),
                            "bbox": [int(x) for x in bbox[0] + bbox[2]] 
                        }
                        print(f"   [+] Found {field_name}: {text}")

        # 4. Contextual Extraction (Bank, Payee, Signature)
        self._extract_contextual_fields(img, valid_results, extracted_data)

        return extracted_data

    def _extract_contextual_fields(self, img, results, data):
        """
        Uses feature-based logic (Size, Keywords) rather than strict positions.
        """
        height, width = img.shape[:2]
        
        # --- A. Smart Bank Name Detection ---
        # Logic: Weighted Score = Box_Area * Position_Weight
        # Top 20% = 3.0x bonus. Top 30% = 1.5x bonus. Below 50% = 0.2x penalty.
        best_bank_score = 0
        best_bank_entry = None
        
        for (bbox, text, prob) in results:
            if "BANK" in text.upper():
                (tl, tr, br, bl) = bbox
                box_h = bl[1] - tl[1]
                box_w = tr[0] - tl[0]
                area = box_h * box_w
                
                # Calculate vertical center (normalized 0.0 to 1.0)
                y_center_norm = ((tl[1] + bl[1]) / 2) / height
                
                # Apply Position Weight
                weight = 1.0
                if y_center_norm < 0.20:
                    weight = 3.0 # Strong preference for top header
                elif y_center_norm < 0.35:
                    weight = 1.5
                elif y_center_norm > 0.50:
                    weight = 0.2 # Penalty for bottom half
                    
                score = area * weight
                
                if score > best_bank_score:
                    best_bank_score = score
                    best_bank_entry = (text, bbox)
        
        if best_bank_entry:
            text, bbox = best_bank_entry
            data["fields"]["BANK_NAME"] = {
                "text": text,
                "bbox": [int(x) for x in bbox[0] + bbox[2]]
            }
            print(f"   [+] Found BANK: {text}")

        # --- B. Smart Payee Detection (Clustering Mode) ---
        # Logic: Find "PAY", define a "Payee Zone" to its right, collect & join all text found there.
        # This handles split handwriting (e.g. "Rahul" "Kumar").
        payee_candidates = []
        stop_words = ["RUPEES", "RS", "OR", "BEARER", "ONLY", "VALID"]
        
        for bbox, text, prob in results:
            if "PAY" in text.upper() and len(text) < 10: # "PAY" or "PAY TO", ignore long noise
                pay_box = bbox
                pay_right_x = pay_box[1][0]
                pay_center_y = (pay_box[0][1] + pay_box[3][1]) / 2
                
                # Scan for words in the "Payee Zone"
                for c_bbox, c_text, c_prob in results:
                    c_left_x = c_bbox[0][0]
                    c_center_y = (c_bbox[0][1] + c_bbox[3][1]) / 2
                    
                    # 1. Must be to the RIGHT of "PAY"
                    if c_left_x > pay_right_x + 10: 
                        # 2. Must be roughly on the same line (Relaxed vertical tolerance for handwriting)
                        if abs(c_center_y - pay_center_y) < 60: 
                            # 3. Must NOT be a stop word
                            if c_text.upper() not in stop_words:
                                payee_candidates.append({
                                    "text": c_text,
                                    "x": c_left_x,
                                    "bbox": [int(x) for x in c_bbox[0] + c_bbox[2]]
                                })
                
                # We only need one "PAY" anchor
                if payee_candidates:
                    break

        if payee_candidates:
            # Sort left-to-right to form a coherent name
            payee_candidates.sort(key=lambda k: k["x"])
            
            # Join text
            full_name = " ".join([c["text"] for c in payee_candidates])
            
            # Use the bounding box of the first and last word to create a super-box
            first_box = payee_candidates[0]["bbox"]
            last_box = payee_candidates[-1]["bbox"]
            merged_bbox = [first_box[0], min(first_box[1], last_box[1]), last_box[2], max(first_box[3], last_box[3])]
            
            data["fields"]["PAYEE_NAME"] = {
                "text": full_name,
                "bbox": merged_bbox
            }
            print(f"   [+] Found PAYEE (Merged): {full_name}")

        # --- C. Smart Signature ROI ---
        # Logic: Find "SIGNATURE" or related keywords. The box is ALWAYS physically above the text.
        # Position agnostic: Wherever the text is, the signature is above it.
        sig_keywords = ["SIGNATURE", "AUTH", "SIGNATORY", "HOLDER", "PLEASE SIGN ABOVE"]
        found_sig = False
        
        for (bbox, text, prob) in results:
            if any(k in text.upper() for k in sig_keywords):
                (tl, tr, br, bl) = bbox
                
                # Define ROI ABOVE the label
                # Widen the box to capture long signatures
                label_width = tr[0] - tl[0]
                roi_w = label_width * 2.5 
                roi_h = height * 0.12 # 12% of cheque height
                
                # Center detection on the label
                roi_x1 = int(tl[0] - (label_width * 0.5))
                roi_x2 = int(roi_x1 + roi_w)
                roi_y2 = int(tl[1]) # Bottom of ROI = Top of Label
                roi_y1 = int(roi_y2 - roi_h)
                
                # Clamp
                roi_x1 = max(0, roi_x1)
                roi_y1 = max(0, roi_y1)
                
                data["fields"]["SIGNATURE_ROI"] = {
                    "text": "[Region Above Label]",
                    "bbox": [roi_x1, roi_y1, roi_x2, roi_y2]
                }
                print(f"   [+] Found SIGNATURE anchor: {text}")
                found_sig = True
                break # Stop after finding the first strong signature label
