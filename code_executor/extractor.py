"""
SAGE Code Executor — Code Extractor
=====================================
Extracts clean Python code from raw LLM responses that may contain:
  • ```python / ```python3 / ```py fenced blocks
  • Generic ``` blocks (no language tag)
  • Prose mixed with code
  • Multiple code blocks (largest wins)
  • Raw code with no fences at all (fallback)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

# Language tags that identify a Python block
_PYTHON_TAGS = frozenset({"python", "python3", "py"})
_PYTHON_PATTERN = r"```(?:python3?|py)\s*\n(.*?)```"
_GENERIC_PATTERN = r"```(?:\w+)?\s*\n(.*?)```"


@dataclass
class CodeExtractResult:
    code: str             # Clean, normalised code string
    language: str         # "python" | "unknown"
    had_fence: bool       # True if a ``` fence was found in the response
    block_count: int      # Number of fenced blocks found (0 if fallback)


def extract_code(raw_response: str) -> CodeExtractResult:
    """
    Extract the best Python code block from an LLM response.

    Priority:
        1. Largest ``python / ```python3 / ```py fenced block
        2. Largest generic ``` fenced block (any or no language tag)
        3. Entire stripped response (fallback — treats whole reply as code)

    Returns:
        CodeExtractResult with normalised code and metadata.
    """
    text = raw_response.strip()

    # ── Priority 1: named Python block ──────────────────────────────────
    python_blocks = _find_blocks(text, _PYTHON_PATTERN)
    if python_blocks:
        best = max(python_blocks, key=len)
        return CodeExtractResult(
            code=_normalise(best),
            language="python",
            had_fence=True,
            block_count=len(python_blocks),
        )

    # ── Priority 2: any fenced block ────────────────────────────────────
    generic_blocks = _find_blocks(text, _GENERIC_PATTERN)
    if generic_blocks:
        best = max(generic_blocks, key=len)
        return CodeExtractResult(
            code=_normalise(best),
            language="unknown",
            had_fence=True,
            block_count=len(generic_blocks),
        )

    # ── Priority 3: treat full response as code ──────────────────────────
    return CodeExtractResult(
        code=_normalise(text),
        language="unknown",
        had_fence=False,
        block_count=0,
    )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _find_blocks(text: str, pattern: str) -> List[str]:
    """Return non-empty stripped code strings matching *pattern*."""
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    return [m.strip() for m in matches if m.strip()]


def _normalise(code: str) -> str:
    """Normalise line endings and strip surrounding whitespace."""
    return code.replace("\r\n", "\n").replace("\r", "\n").strip()
