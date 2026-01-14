# Deceptive Review Detection

**Author & Copyright © 2026 Kavindu Hansaka Jayasinghe**  
All rights reserved.

---

## 📌 Overview

This project implements a **semantic deceptive review detection system** that identifies **internal contradictions within a single product review**.

The solution follows a **hybrid NLP + LLM approach**, combining:
- Sentence-level semantic embeddings for efficient candidate selection
- Lightweight clustering and polarity analysis
- **Granite 4.0 h Tiny LLM (GGUF, local inference, compulsory)** for final contradiction reasoning

The system detects **explicit, implicit, and expectation-level contradictions**, fully aligned with Twist Digital’s problem definition.

---

## 🎯 Problem Alignment

This implementation directly addresses the Twist Digital technical assignment requirements:

- ✅ Detects contradictions *within a single review*
- ✅ Produces structured outputs (flag, confidence, explanation)
- ✅ Uses modern NLP / LLM-based techniques
- ✅ Goes beyond rule-based or classical NLP limitations
- ✅ Demonstrates clear architectural reasoning

---

## 🏗️ System Architecture

```
Input Review
   ↓
Sentence Segmentation
   ↓
Sentence Embeddings (MiniLM)
   ↓
Semantic Clustering (DBSCAN)
   ↓
Polarity Estimation (Positive vs Negative Anchors)
   ↓
Candidate Pair Filtering
   ↓
LLM Reasoning
   ↓
Contradiction Flag + Confidence + Explanation
```

Granite is **always executed** before producing a final decision, satisfying the compulsory LLM requirement.

---

## 🤖 Large Language Model 

### Granite LLM

- **Base model**:  
  https://huggingface.co/ibm-granite/granite-4.0-h-tiny
- **Execution format**: GGUF (Q4_0)
- **GGUF conversion**: Created by the author ([KavinduHansaka/granite-4.0-h-tiny-gguf](https://huggingface.co/KavinduHansaka/granite-4.0-h-tiny-gguf))
- **Runtime**: Local inference using `llama-cpp-python`

Granite is used for **semantic reasoning only**, not text generation, ensuring deterministic and explainable behavior.

---

## 🧩 Design Rationale (Rubric-Oriented)

### Why not classical NLP only?
Classical NLP techniques struggle with:
- Implicit contradictions
- Multi-sentence logical conflicts
- Expectation-level inconsistencies

LLM-based reasoning is required to address these reliably.

### Why a hybrid approach?
- Embeddings → reduce computational cost
- Granite → authoritative semantic judgment

This balances **accuracy, performance, and scalability**.

---

## ⚙️ Features

- Detects explicit and implicit contradictions
- Handles expectation-level inconsistencies
- Produces human-readable explanations
- Confidence scoring (0–1)
- Fully local LLM execution
- REST API (FastAPI)
- Simple Tailwind-based UI
- Batch testing support

---

## 📂 Project Structure

```
deceptive-review-detection/
│
├── app.py
├── README.md
├── requirements.txt
├── test_reviews.py
│
├── src/
│   └── detector.py
│
├── models/
│   └── granite-4.0-h-tiny-Q4_0.gguf
│
├── frontend/
│   └── index.html
│
├── data/
    └── dataset.txt

```

---

## 🚀 How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Place Granite GGUF model
```
models/granite-4.0-h-tiny-Q4_0.gguf
```

### Start the server
```bash
uvicorn app:app --reload
```

### Open UI
```
http://127.0.0.1:8000
```

---

## 🧪 Testing

```bash
python test_reviews.py
```

Sample output:
```
❌ Contradiction: YES
Confidence: 0.72
Explanation:
Granite detected a logical inconsistency between durability and failure claims.
```

---

## ⚠️ Limitations

- Subtle sarcasm may remain ambiguous
- Confidence is heuristic-based
- Requires sufficient textual content

These limitations are inherent to natural language understanding tasks.

---

## 🛡️ License & Attribution

© 2026 Kavindu Jayasinghe. All rights reserved.

- Codebase: Copyrighted by the author
- Granite base model: Apache-2.0 (IBM)
- GGUF conversion: Performed by the author
- GGUF redistribution subject to IBM Granite licensing

---
