# Semi-Supervised Document Field Extraction with Curriculum Learning: A Reinforcement Learning Approach for Indian Bank Cheques

> **MCA Major Project (2026)**  
> **Status:** Phase 1 (Baseline) Completed ✅

## 📌 Executive Summary
This project aims to solve the challenge of extracting structured data (IFSC, MICR, Account No, Payee, etc.) from Indian Bank Cheques using minimal labeled data. 

**Core Innovation:** Combining **Self-Supervised Pre-training (LayoutLMv3)** with a **Reinforcement Learning (RL) Agent** that learns to "hunt" for fields in complex, noisy layouts without explicit coordinate supervision.

---

## 🚀 Setup & Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/MegtonLabs/Major-Project.git
   cd Major-Project
   ```

2. **Create Environment**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Install PyTorch with CUDA support manually if needed.*

---

## 🏃‍♂️ Usage

### Quick Test (Sandbox)
To test the extraction pipeline on a random cheque from your dataset:

```bash
python run_sandbox_test.py
```
*   **Output:** Check `testing_sandbox/debug_output.jpg` for visualization.

### Test Specific Image
```bash
python run_sandbox_test.py "path/to/image.jpg"
```

---

## 📂 Project Structure

```text
├── data/               # Dataset (Not on GitHub)
├── models/             # Cached Weights (LayoutLMv3/EasyOCR - Local only)
├── src/
│   ├── agents/         # Extraction Logic (Baseline & RL)
│   ├── data/           # Smart Loaders (Orientation AI)
│   └── models/         # Neural Network Definitions
├── experiments/        # Research Logs
└── docs/               # Project Documentation
```

## 📅 Roadmap

- [x] **Week 1-2:** Environment Setup & Rule-Based Baseline (Accuracy: ~70%)
- [ ] **Week 3-4:** Self-Supervised Pre-training (LayoutLMv3)
- [ ] **Week 5:** RL "Hunter" Agent (PPO)
- [ ] **Week 6:** Curriculum Learning Integration
- [ ] **Week 7-8:** Final Analysis & Thesis

---
*Developed by Megton Labs*
