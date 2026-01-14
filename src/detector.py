"""
Semantic Contradiction Detector
LLM-based reasoning is compulsory internally.
User-facing output is model-agnostic and clean.
"""

import re
import itertools
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

from llama_cpp import Llama


# =========================
# Configuration
# =========================
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.72
CLUSTER_EPS = 0.25

POSITIVE_ANCHOR = "This works very well and is high quality"
NEGATIVE_ANCHOR = "This works very badly and is poor quality"


# =========================
# Data Structures
# =========================
@dataclass
class Claim:
    sentence: str
    embedding: np.ndarray
    cluster_id: int
    polarity: float


@dataclass
class ContradictionResult:
    has_contradiction: bool
    confidence: float
    contradicting_pairs: List[Tuple[str, str]]
    explanation: str


# =========================
# Sanitization
# =========================
def clean_explanation(text: str) -> str:
    if not text:
        return ""

    stop_markers = [
        "def test",
        "self.assert",
        "unittest",
        "pytest",
        "example:",
        "```",
        "class ",
        "import ",
    ]

    lowered = text.lower()
    for marker in stop_markers:
        idx = lowered.find(marker)
        if idx != -1:
            text = text[:idx]
            lowered = text.lower()

    return text.strip()


def parse_answer(text: str) -> bool:
    """
    Extracts YES / NO decision from structured LLM output.
    """
    for line in text.splitlines():
        if line.lower().startswith("answer:"):
            return "yes" in line.lower()
    return False


# =========================
# Detector
# =========================
class SemanticContradictionDetector:

    def __init__(self, granite_model_path: str):
        print("🔹 Loading embedding model...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        print("🔹 Loading LLM model...")
        self.llm = Llama(
            model_path=granite_model_path,
            n_ctx=2048,
            temperature=0.0,
        )

        self.pos_anchor = self.embedder.encode(POSITIVE_ANCHOR)
        self.neg_anchor = self.embedder.encode(NEGATIVE_ANCHOR)

        #print("✅ LLM ENABLED (COMPULSORY)\n")

    # -------------------------
    # Sentence splitting
    # -------------------------
    def split_sentences(self, text: str) -> List[str]:
        text = re.sub(r"\s+", " ", text.strip())
        for sep in [".", "!", "?"]:
            text = text.replace(sep, "|")
        return [s.strip() for s in text.split("|") if len(s.strip()) > 5]

    # -------------------------
    # Polarity scoring
    # -------------------------
    def polarity_score(self, embedding: np.ndarray) -> float:
        pos = cosine_similarity([embedding], [self.pos_anchor])[0][0]
        neg = cosine_similarity([embedding], [self.neg_anchor])[0][0]
        return pos - neg

    # -------------------------
    # LLM: Pairwise check
    # -------------------------
    def llm_pair_check(
        self, a: str, b: str, polarity_gap: float
    ) -> Tuple[bool, float]:

        self.llm.reset()

        prompt = f"""
            You are checking all statements from the SAME product review.

            A contradiction exists if:
            - One statement claims a negative quality
            - Another claims a positive outcome
            - About the same aspect
            - And both cannot logically be true together

            Statement A:
            "{a}"

            Statement B:
            "{b}"

            Respond EXACTLY in this format:

            Answer: YES or NO
            Reason:
            <one short explanation>

            Do NOT include code, tests, or examples.
            """

        response = self.llm(prompt, max_tokens=160)
        raw = response["choices"][0]["text"]

        cleaned = clean_explanation(raw)
        is_contra = parse_answer(cleaned)
        confidence = min(0.95, 0.5 + polarity_gap)

        return is_contra, confidence

    # -------------------------
    # LLM: Global check
    # -------------------------
    def llm_global_check(self, text: str) -> Tuple[bool, float]:

        self.llm.reset()

        prompt = f"""
            You are analyzing a single product review.

            A contradiction exists if:
            - One statement claims a negative quality
            - Another statement claims a positive outcome
            - About the same product or service aspect
            - And they cannot logically both be true

            Examples:
            - "Unhelpful support" vs "Issue resolved quickly with a discount"
            - "Very durable" vs "Broke on first use"

            Review:
            {text}

            Respond EXACTLY in this format:

            Answer: YES or NO
            Reason:
            <one short explanation>

            Do NOT include code, tests, or examples.
            """

        response = self.llm(prompt, max_tokens=200)
        raw = response["choices"][0]["text"]

        cleaned = clean_explanation(raw)
        is_contra = parse_answer(cleaned)
        confidence = 0.65 if is_contra else 0.35

        return is_contra, confidence

    # -------------------------
    # Main pipeline
    # -------------------------
    def analyze(self, text: str) -> ContradictionResult:

        sentences = self.split_sentences(text)
        if len(sentences) < 2:
            return ContradictionResult(
                False, 0.2, [], "Not enough content to analyze contradictions."
            )

        embeddings = self.embedder.encode(sentences)
        clusters = DBSCAN(
            eps=CLUSTER_EPS,
            min_samples=1,
            metric="cosine"
        ).fit(embeddings).labels_

        claims: List[Claim] = []
        for s, e, c in zip(sentences, embeddings, clusters):
            claims.append(
                Claim(
                    sentence=s,
                    embedding=e,
                    cluster_id=int(c),
                    polarity=self.polarity_score(e),
                )
            )

        contradicting_pairs = []
        confidences = []

        # Pairwise checks
        for a, b in itertools.combinations(claims, 2):
            if a.cluster_id != b.cluster_id:
                continue

            sim = cosine_similarity([a.embedding], [b.embedding])[0][0]
            if sim < SIMILARITY_THRESHOLD:
                continue

            gap = abs(a.polarity - b.polarity)
            if gap < 0.35:
                continue

            is_contra, conf = self.llm_pair_check(a.sentence, b.sentence, gap)

            if is_contra:
                contradicting_pairs.append((a.sentence, b.sentence))
                confidences.append(conf)

        # Global fallback
        if not contradicting_pairs:
            is_global, conf = self.llm_global_check(text)
            return ContradictionResult(
                is_global,
                conf,
                [],
                "Detected a contradiction." if is_global else "No contradiction detected."
            )

        return ContradictionResult(
            True,
            float(np.mean(confidences)),
            contradicting_pairs,
            "Detected a contradiction between multiple claims."
        )
