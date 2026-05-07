# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Skill loading utilities for agents."""

import shlex
from pathlib import Path
from typing import Any, Iterable

from pydantic_ai.messages import ModelMessage, ModelRequest, SystemPromptPart

DEFAULT_TASK_SKILL_ROOT = "/testbed"
DEFAULT_SKILL_FILENAME = "SKILL.md"


def _resolve_skill_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_dir():
        expanded = expanded / DEFAULT_SKILL_FILENAME
    return expanded


def _format_skill_instructions(sections: Iterable[tuple[str, str]]) -> str | None:
    rendered_sections = [
        f'<skill path="{source}">\n{content.strip()}\n</skill>'
        for source, content in sections
        if content.strip()
    ]
    if not rendered_sections:
        return None

    return (
        "The following skills are available. Use them when they are relevant to the task.\n\n"
        "<skills>\n"
        + "\n\n".join(rendered_sections)
        + "\n</skills>"
    )


def load_skill_instructions(skill_paths: Iterable[Path]) -> str | None:
    """Load host skill files and render them as system instructions."""
    sections: list[tuple[str, str]] = []
    for raw_path in skill_paths:
        path = _resolve_skill_path(raw_path)
        content = path.read_text(encoding="utf-8")
        sections.append((str(path), content))

    return _format_skill_instructions(sections)


def append_skill_message(
    message_history: Iterable[ModelMessage], skill_paths: Iterable[Path]
) -> list[ModelMessage]:
    """Append loaded skill instructions as a system message."""
    messages = list(message_history)
    skill_instructions = load_skill_instructions(skill_paths)
    if skill_instructions is not None:
        messages.append(ModelRequest(parts=[SystemPromptPart(skill_instructions)]))
    return messages


async def load_task_root_skill_instructions(
    env_state: Any,
    *,
    root: str = DEFAULT_TASK_SKILL_ROOT,
    filename: str = DEFAULT_SKILL_FILENAME,
) -> str | None:
    """Load a skill file from the running task container root, when available."""
    sandbox_manager = getattr(env_state, "sandbox_manager", None)
    agent_container_id = getattr(env_state, "agent_container_id", None)
    if sandbox_manager is None or agent_container_id is None:
        return None

    root = root.rstrip("/") or "/"
    skill_path = f"{root}/{filename}" if root != "/" else f"/{filename}"
    quoted_skill_path = shlex.quote(skill_path)
    result = await sandbox_manager.exec(
        agent_container_id,
        ["sh", "-c", f"if [ -f {quoted_skill_path} ]; then cat {quoted_skill_path}; fi"],
        cwd=root,
        timeout=5,
    )
    if getattr(result, "exit_code", 1) != 0:
        return None

    content = getattr(result, "stdout", None) or ""
    return _format_skill_instructions([(f"container:{skill_path}", content)])


async def append_task_root_skill_message(
    message_history: Iterable[ModelMessage],
    env_state: Any,
) -> list[ModelMessage]:
    """Append the running task root skill as a system message, if present."""
    messages = list(message_history)
    skill_instructions = await load_task_root_skill_instructions(env_state)
    if skill_instructions is not None:
        messages.append(ModelRequest(parts=[SystemPromptPart(skill_instructions)]))
    return messages
