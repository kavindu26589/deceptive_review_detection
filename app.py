from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.detector import SemanticContradictionDetector

app = FastAPI(title="Deceptive Review Detection")

detector = SemanticContradictionDetector(
    granite_model_path="models/granite-4.0-h-tiny-Q4_0.gguf"
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("frontend/index.html", "r", encoding="utf-8") as f:
        return f.read()

class ReviewRequest(BaseModel):
    text: str

@app.post("/analyze")
def analyze_review(req: ReviewRequest):
    result = detector.analyze(req.text)
    return {
        "has_contradiction": result.has_contradiction,
        "confidence": result.confidence,
        "pairs": result.contradicting_pairs,
        "explanation": result.explanation,
    }
