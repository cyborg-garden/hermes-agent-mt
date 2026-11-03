"""Boot.md plugin -- runs BOOT.md as a startup prompt on first session.

Reads ~/.hermes/BOOT.md on the first session start and sends it as
a one-shot agent prompt in a background thread. If nothing needs
attention, the agent replies with [SILENT] and the message is suppressed.
"""

import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


def _get_boot_file() -> Path:
    """Get the BOOT.md path from HERMES_HOME."""
    from hermes_cli.config import get_hermes_home
    return get_hermes_home() / "BOOT.md"


_boot_executed = False


def _build_boot_prompt(content: str) -> str:
    """Wrap BOOT.md content in a system-level instruction."""
    return (
        "You are running a startup boot checklist. Follow the BOOT.md "
        "instructions below exactly.\n\n"
        "---\n"
        f"{content}\n"
        "---\n\n"
        "Execute each instruction. If you need to send a message to a "
        "platform, use the send_message tool.\n"
        "If nothing needs attention and there is nothing to report, "
        "reply with ONLY: [SILENT]"
    )


def _run_boot_agent(content: str) -> None:
    """Spawn a one-shot agent session to execute the boot instructions."""
    try:
        from run_agent import AIAgent

        prompt = _build_boot_prompt(content)
        agent = AIAgent(
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            max_iterations=20,
        )
        result = agent.run_conversation(prompt)
        response = result.get("final_response", "")
        if response and "[SILENT]" not in response:
            logger.info("boot-md completed: %s", response[:200])
        else:
            logger.info("boot-md completed (nothing to report)")
    except Exception as e:
        logger.error("boot-md agent failed: %s", e)


def _on_session_start(session_id: str = "", **kwargs) -> None:
    """Run BOOT.md on first session start only."""
    global _boot_executed
    if _boot_executed:
        return
    _boot_executed = True

    boot_file = _get_boot_file()
    if not boot_file.exists():
        return

    content = boot_file.read_text(encoding="utf-8-sig").strip()
    if not content:
        return

    logger.info("Running BOOT.md (%d chars)", len(content))
    thread = threading.Thread(
        target=_run_boot_agent,
        args=(content,),
        name="boot-md",
        daemon=True,
    )
    thread.start()


def register(ctx) -> None:
    """Register boot-md as an on_session_start hook."""
    ctx.register_hook("on_session_start", _on_session_start)
    logger.info("boot-md: registered (triggers on first session start)")
