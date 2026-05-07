# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Skill loading utilities for agents."""

from pathlib import Path
from typing import Iterable

from pydantic_ai.messages import ModelMessage, ModelRequest, SystemPromptPart


def _resolve_skill_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_dir():
        expanded = expanded / "SKILL.md"
    return expanded


def load_skill_instructions(skill_paths: Iterable[Path]) -> str | None:
    """Load skill files and render them as system instructions."""
    sections: list[str] = []
    for raw_path in skill_paths:
        path = _resolve_skill_path(raw_path)
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue
        sections.append(f"<skill path=\"{path}\">\n{content}\n</skill>")

    if not sections:
        return None

    return (
        "The following skills are available. Use them when they are relevant to the task.\n\n"
        "<skills>\n"
        + "\n\n".join(sections)
        + "\n</skills>"
    )


def append_skill_message(
    message_history: Iterable[ModelMessage], skill_paths: Iterable[Path]
) -> list[ModelMessage]:
    """Append loaded skill instructions as a system message."""
    messages = list(message_history)
    skill_instructions = load_skill_instructions(skill_paths)
    if skill_instructions is not None:
        messages.append(ModelRequest(parts=[SystemPromptPart(skill_instructions)]))
    return messages
