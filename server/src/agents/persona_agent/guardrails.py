from __future__ import annotations
import re
from src.models.retrieval import RetrievedChunk
from src.config.tuning import retrieval as cfg

_INJECTION_PATTERNS = [
    r"ignore (previous|prior|all|the above) instructions",
    r"disregard (previous|prior|all|the above)",
    r"forget (everything|what you were told|your instructions)",
    r"you are now (a |an )?(?!vansh)",
    r"act as (a |an )?(?!vansh)",
    r"pretend (you are|to be)",
    r"reveal (your|the) (system )?prompt",
    r"show me (your|the) (system )?prompt",
    r"what (are|were) your instructions",
    r"override (your|the) (system |original )?instructions",
    r"new (persona|role|identity)",
    r"jailbreak",
    r"DAN mode",
]

_INJECTION_RE = re.compile(
    "|".join(_INJECTION_PATTERNS),
    re.IGNORECASE,
)

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

MAX_INPUT_CHARS = 2048


def detect_injection(text: str) -> bool:
    return bool(_INJECTION_RE.search(text))


def check_grounding(chunks: list[RetrievedChunk]) -> bool:
    """
    Returns True if retrieval is grounded.
    Uses cosine similarity score (stored in metadata.cos_score) when available,
    since RRF fusion scores are not calibrated to the [0,1] cosine range.
    """
    if not chunks:
        return False
    top_score = max(
        c.metadata.get("cos_score", c.score) for c in chunks
    )
    return top_score >= cfg.CONFIDENCE_THRESHOLD


def sanitize_input(text: str) -> str:
    text = _CONTROL_CHAR_RE.sub("", text)
    return text[:MAX_INPUT_CHARS].strip()
