import easyocr
import sys

# Force UTF-8 for stdout
sys.stdout.reconfigure(encoding='utf-8')

print("📥 Pre-downloading EasyOCR models...")
reader = easyocr.Reader(['en'], gpu=True)
print("✅ EasyOCR models ready.")
