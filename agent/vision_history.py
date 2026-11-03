"""What a stored vision result may say.

The pixels are never saved, so a row reloaded from the database has none. A
sentence telling the model it can see them is only true beside the pixels.
"""

from __future__ import annotations

import re
from typing import Any, Optional

# Sits next to the pixels in the tool result. Only true while they travel with it.
NATIVE_VISION_ACK = (
    "Image loaded into your context — you can see it natively now. "
    "Use your built-in vision to answer the user."
)
IMAGE_NOT_IN_HISTORY = (
    "Image from an earlier turn. Its pixels are not kept in the history, so you "
    "cannot see it now. Do not describe it from this row. "
    "Use vision_analyze if you need to look again."
)
# Stored by older code, so these rows are still in session databases.
_OLD_SUMMARY_RE = re.compile(
    r"Image attached natively for the main model \([^)]*\)\. Answer using built-in vision\."
)
# Other tools can quote these sentences on purpose, such as a memory note about them.
_VISION_TOOLS = frozenset({"vision_analyze", "browser_vision"})


def correct_stored_vision_row(tool_name: Optional[str], content: Any) -> Any:
    """Return a tool row's text with any claim of sight replaced by the truth."""
    if tool_name not in _VISION_TOOLS:
        return content
    if isinstance(content, list):
        # In-place compaction can store a live multimodal row whole.
        return [
            {**part, "text": _corrected_text(part["text"])}
            if isinstance(part, dict) and isinstance(part.get("text"), str)
            else part
            for part in content
        ]
    return _corrected_text(content) if isinstance(content, str) else content


def _corrected_text(text: str) -> str:
    return _OLD_SUMMARY_RE.sub(
        IMAGE_NOT_IN_HISTORY, text.replace(NATIVE_VISION_ACK, IMAGE_NOT_IN_HISTORY)
    )


def snippet_of_corrected_row(snippet: Optional[str], corrected: str, limit: int = 240) -> str:
    """Return a short snippet of a corrected row, near what the search matched.

    The search engine's own snippet came from the uncorrected text, so it may
    quote the claim. Whole rows can run to tens of thousands of characters.
    """
    matched = re.search(r">>>(.*?)<<<", snippet or "")
    at = corrected.lower().find(matched.group(1).lower()) if matched else -1
    if at < 0:
        at = corrected.find(IMAGE_NOT_IN_HISTORY)
    start = max(at - limit // 3, 0)
    end = start + limit
    return f"{'...' if start else ''}{corrected[start:end]}{'...' if end < len(corrected) else ''}"
