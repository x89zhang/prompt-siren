# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Data models for job management and persistence."""

from __future__ import annotations

import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelRequestPart
from pydantic_ai.usage import RunUsage

from ..config.experiment_config import ExperimentConfig
from ..types import ExecutionMode, InjectableModelRequestPart


class ExceptionInfo(BaseModel):
    """Information about an exception that occurred during task execution."""

    exception_type: str
    exception_message: str
    exception_traceback: str
    occurred_at: datetime

    @classmethod
    def from_exception(cls, e: BaseException) -> ExceptionInfo:
        """Create ExceptionInfo from an exception."""
        return cls(
            exception_type=type(e).__name__,
            exception_message=str(e),
            exception_traceback=traceback.format_exc(),
            occurred_at=datetime.now(),
        )


class TaskRunResult(BaseModel):
    """Result for a single task run (one execution of a task).

    A task may be run multiple times for pass@k evaluation.
    """

    task_id: str
    run_id: str
    started_at: datetime | None = None
    finished_at: datetime | None = None
    benign_score: float | None = None
    attack_score: float | None = None
    trajectory_level: Literal["L0", "L1", "L2", "L3", "L4", "L5"] | None = None
    classification_method: (
        Literal["hybrid", "judge_audit", "attack_relation", "attack_chain_judge"] | None
    ) = None
    attacks: dict[str, Any] | None = None
    exception_info: ExceptionInfo | None = None


class TaskRunExecution(BaseModel):
    """Full execution data for a task run."""

    task_id: str
    run_id: str
    execution_id: str
    timestamp: datetime
    trace_id: str | None = None
    span_id: str | None = None
    messages: list[ModelMessage]
    usage: RunUsage
    attacks: dict[str, Any] | None = None  # Generated attacks
    trajectory_labels: dict[str, Any] | None = None
    attack_analysis: dict[str, Any] | None = None
    attack_chain_judgment: dict[str, Any] | None = None
    resume_state: TaskRunResumeState | None = None


class TaskRunExecutionMetadata(BaseModel):
    """Analysis metadata associated with a task run execution."""

    task_id: str
    run_id: str
    execution_id: str
    timestamp: datetime
    trace_id: str | None = None
    span_id: str | None = None
    attacks: dict[str, Any] | None = None
    trajectory_labels: dict[str, Any] | None = None
    attack_analysis: dict[str, Any] | None = None
    attack_chain_judgment: dict[str, Any] | None = None


class TaskRunResumeState(BaseModel):
    """Serializable state-machine checkpoint for resuming an in-progress run."""

    state_kind: Literal["model_request", "injectable_model_request", "model_response", "end"]
    model_request: ModelRequest | None = None
    injectable_model_request_parts: list[ModelRequestPart | InjectableModelRequestPart] | None = (
        None
    )


class ToolReplayEntry(BaseModel):
    """One completed historical tool call replayed before resuming a checkpoint."""

    message_index: int
    tool_call_id: str | None = None
    tool_name: str
    args: dict[str, Any]
    replayed_at: datetime
    outcome: Literal["success", "retry", "error"]
    result_preview: str | None = None
    error_type: str | None = None
    error_message: str | None = None


class TaskRunToolReplay(BaseModel):
    """Audit record for tool-history replay performed before partial resume."""

    task_id: str
    run_id: str
    source_run_id: str
    source_execution_id: str
    replayed_at: datetime
    entries: list[ToolReplayEntry]


class ToolHistoryEntry(BaseModel):
    """Tool call extracted from execution message history."""

    message_index: int
    tool_call_id: str | None = None
    tool_name: str
    args: dict[str, Any]
    completed: bool
    return_message_index: int | None = None


class TaskRunToolHistory(BaseModel):
    """Tool-call history extracted from an execution trace."""

    task_id: str
    run_id: str
    execution_id: str
    timestamp: datetime
    entries: list[ToolHistoryEntry]


class JobConfig(ExperimentConfig):
    """Snapshot of the experiment configuration for a job.

    Extends ExperimentConfig with job-specific metadata. This is saved as
    config.yaml in the job directory and can be loaded via OmegaConf for
    resume with Hydra-style overrides.
    """

    job_name: str
    execution_mode: ExecutionMode
    created_at: datetime
    n_runs_per_task: int = Field(default=1, ge=1, description="Number of runs per task for pass@k")


class RunIndexEntry(BaseModel):
    """Entry in per-job index.jsonl for fast result lookup."""

    task_id: str
    run_id: str  # 8-char UUID
    timestamp: datetime
    benign_score: float | None
    attack_score: float | None
    trajectory_level: Literal["L0", "L1", "L2", "L3", "L4", "L5"] | None = None
    classification_method: (
        Literal["hybrid", "judge_audit", "attack_relation", "attack_chain_judge"] | None
    ) = None
    exception_type: str | None = None  # None if successful
    path: Path  # Relative path to run directory from job directory


# Constants for file names
CONFIG_FILENAME = "config.yaml"
INDEX_FILENAME = "index.jsonl"
ASR_FILENAME = "asr.json"
RESULT_FILENAME = "result.json"
INDEX_LOCK_FILENAME = "index.jsonl.lock"
TASK_RESULT_FILENAME = "result.json"
TASK_EXECUTION_FILENAME = "execution.json"
TASK_EXECUTION_METADATA_FILENAME = "execution_metadata.json"
TASK_ATTACK_CHAIN_FILENAME = "attack_chain.json"
TASK_ATTACK_ANALYSIS_FILENAME = "attack_analysis_units.json"
TASK_ATTACK_CHAIN_JUDGE_FILENAME = "attack_chain_judge.json"
TASK_ATTACK_CHAIN_JUDGE_MARKDOWN_FILENAME = "attack_chain_judge.md"
TASK_TOOL_REPLAY_FILENAME = "tool_replay.json"
TASK_TOOL_HISTORY_FILENAME = "tool_history.json"
