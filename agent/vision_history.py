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
    if tool_name not in _VISION_TOOLS or not isinstance(content, str):
        return content
    return _OLD_SUMMARY_RE.sub(
        IMAGE_NOT_IN_HISTORY, content.replace(NATIVE_VISION_ACK, IMAGE_NOT_IN_HISTORY)
    )
