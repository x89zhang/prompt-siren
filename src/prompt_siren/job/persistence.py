# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Persistence layer for job-based task execution results."""

from __future__ import annotations

import json
import uuid
from collections import Counter
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

import logfire
import yaml
from filelock import FileLock
from logfire import LogfireSpan
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.usage import RunUsage

from ..attack_success_cases import record_attack_success_case
from ..tasks import EvaluationResult, Task, TaskCouple
from ..trajectory_labeling import label_trajectory, LEVEL_RANK, TrajectoryLabels, UptakeLevel
from ..types import (
    InjectableModelMessagesTypeAdapter,
    InjectableToolReturnPart,
    InjectionAttack,
    InjectionAttacksDictTypeAdapter,
)
from .models import (
    CONFIG_FILENAME,
    ExceptionInfo,
    INDEX_FILENAME,
    JobConfig,
    RunIndexEntry,
    TASK_ATTACK_CHAIN_FILENAME,
    TASK_EXECUTION_FILENAME,
    TASK_EXECUTION_METADATA_FILENAME,
    TASK_RESULT_FILENAME,
    TASK_TOOL_HISTORY_FILENAME,
    TASK_TOOL_REPLAY_FILENAME,
    TaskRunExecution,
    TaskRunExecutionMetadata,
    TaskRunResult,
    TaskRunResumeState,
    TaskRunToolHistory,
    TaskRunToolReplay,
    ToolHistoryEntry,
)
from .naming import sanitize_for_filename

EnvStateT = TypeVar("EnvStateT")


class JobPersistence:
    """Manages persistence for a single job's task executions.

    Directory structure:
        job_dir/
            config.yaml           # Job configuration
            index.jsonl           # Index of all task runs
            <task_id>/
                <run_id>/            # 8-char UUID
                    result.json      # Task run result
                    execution.json   # Raw execution trace and checkpoint
                    execution_metadata.json  # Attacks and trajectory labels
                    attack_chain.json  # Non-L0 messages extracted from trace
                <run_id>/
                    ...
    """

    def __init__(self, job_dir: Path, job_config: JobConfig):
        """Initialize JobPersistence.

        Args:
            job_dir: Directory for this job's data
            job_config: Configuration for this job
        """
        self.job_dir = job_dir
        self.job_config = job_config
        self.resume_partial_in_place = False
        self.resume_only_partial = False
        self.resume_partial_repeats = 1
        self.resume_replay_tool_history = True

    @classmethod
    def create(cls, job_dir: Path, job_config: JobConfig) -> JobPersistence:
        """Create a new JobPersistence and initialize the job directory.

        Args:
            job_dir: Directory for this job's data
            job_config: Configuration for this job

        Returns:
            JobPersistence instance with initialized directory
        """
        job_dir.mkdir(parents=True, exist_ok=True)

        # Save config.yaml if it doesn't exist
        config_path = job_dir / CONFIG_FILENAME
        if not config_path.exists():
            _save_config_yaml(config_path, job_config)

        return cls(job_dir, job_config)

    def get_task_run_dir(self, task_id: str, run_id: str) -> Path:
        """Get the directory path for a specific task run.

        Args:
            task_id: Task identifier
            run_id: Run identifier (8-char UUID)

        Returns:
            Path to the task run directory
        """
        task_id_safe = sanitize_for_filename(task_id)
        return self.job_dir / task_id_safe / run_id

    def create_task_run_dir(self, task_id: str) -> tuple[str, Path]:
        """Create and return a stable run ID and directory for an in-progress run."""
        run_id = str(uuid.uuid4())[:8]
        run_dir = self.get_task_run_dir(task_id, run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_id, run_dir

    def save_partial_execution(
        self,
        *,
        task_id: str,
        run_id: str,
        messages: list[ModelMessage],
        usage: RunUsage,
        task_span: LogfireSpan,
        generated_attacks: Mapping[str, InjectionAttack] | None = None,
        resume_state: TaskRunResumeState | None = None,
    ) -> Path:
        """Save the latest known execution state without updating result/index files."""
        run_dir = self.get_task_run_dir(task_id, run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now()
        execution_id = str(uuid.uuid4())[:8]

        span_context = task_span.get_span_context() if task_span is not None else None
        trace_id = format(span_context.trace_id, "032x") if span_context else None
        span_id = format(span_context.span_id, "016x") if span_context else None

        attacks_dict: dict[str, Any] | None = None
        if generated_attacks:
            attacks_dict = InjectionAttacksDictTypeAdapter.dump_python(dict(generated_attacks))

        execution = TaskRunExecution(
            task_id=task_id,
            run_id=run_id,
            execution_id=execution_id,
            timestamp=now,
            trace_id=trace_id,
            span_id=span_id,
            messages=messages,
            usage=usage,
            attacks=attacks_dict,
            resume_state=resume_state,
        )
        self._save_execution_files(
            run_dir,
            execution,
            attacks=attacks_dict,
            trajectory_labels=None,
        )
        return run_dir

    def load_execution(self, task_id: str, run_id: str) -> TaskRunExecution:
        """Load execution data for a task run."""
        run_dir = self.get_task_run_dir(task_id, run_id)
        execution_path = run_dir / TASK_EXECUTION_FILENAME
        execution = TaskRunExecution.model_validate_json(execution_path.read_text())

        metadata_path = run_dir / TASK_EXECUTION_METADATA_FILENAME
        if not metadata_path.exists():
            return execution

        metadata = TaskRunExecutionMetadata.model_validate_json(metadata_path.read_text())
        return execution.model_copy(
            update={
                "attacks": metadata.attacks,
                "trajectory_labels": metadata.trajectory_labels,
            }
        )

    def save_tool_replay(self, replay: TaskRunToolReplay) -> Path:
        """Save tool-history replay audit data for a resumed run."""
        run_dir = self.get_task_run_dir(replay.task_id, replay.run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        replay_path = run_dir / TASK_TOOL_REPLAY_FILENAME
        tmp_path = replay_path.with_suffix(f"{replay_path.suffix}.tmp")
        with open(tmp_path, "w") as f:
            f.write(replay.model_dump_json(indent=2))
        tmp_path.replace(replay_path)
        return replay_path

    def _save_tool_history_file(self, run_dir: Path, execution: TaskRunExecution) -> None:
        history = TaskRunToolHistory(
            task_id=execution.task_id,
            run_id=execution.run_id,
            execution_id=execution.execution_id,
            timestamp=execution.timestamp,
            entries=_extract_tool_history_entries(execution.messages),
        )
        history_path = run_dir / TASK_TOOL_HISTORY_FILENAME
        tmp_path = history_path.with_suffix(f"{history_path.suffix}.tmp")
        with open(tmp_path, "w") as f:
            f.write(history.model_dump_json(indent=2))
        tmp_path.replace(history_path)

    def _save_execution_files(
        self,
        run_dir: Path,
        execution: TaskRunExecution,
        *,
        attacks: dict[str, Any] | None,
        trajectory_labels: dict[str, Any] | None,
    ) -> None:
        execution_path = run_dir / TASK_EXECUTION_FILENAME
        tmp_path = execution_path.with_suffix(f"{execution_path.suffix}.tmp")
        with open(tmp_path, "w") as f:
            f.write(
                execution.model_dump_json(
                    indent=2,
                    exclude={"attacks", "trajectory_labels"},
                )
            )
        tmp_path.replace(execution_path)
        self._save_tool_history_file(run_dir, execution)

        metadata_path = run_dir / TASK_EXECUTION_METADATA_FILENAME
        if attacks is None and trajectory_labels is None:
            return

        metadata = TaskRunExecutionMetadata(
            task_id=execution.task_id,
            run_id=execution.run_id,
            execution_id=execution.execution_id,
            timestamp=execution.timestamp,
            trace_id=execution.trace_id,
            span_id=execution.span_id,
            attacks=attacks,
            trajectory_labels=trajectory_labels,
        )
        tmp_metadata_path = metadata_path.with_suffix(f"{metadata_path.suffix}.tmp")
        with open(tmp_metadata_path, "w") as f:
            f.write(metadata.model_dump_json(indent=2))
        tmp_metadata_path.replace(metadata_path)

        if trajectory_labels is not None:
            self._save_attack_chain_file(run_dir, execution, trajectory_labels)

    def _save_attack_chain_file(
        self,
        run_dir: Path,
        execution: TaskRunExecution,
        trajectory_labels: dict[str, Any],
    ) -> None:
        attack_chain = {
            "task_id": execution.task_id,
            "run_id": execution.run_id,
            "execution_id": execution.execution_id,
            "timestamp": execution.timestamp.isoformat(),
            "trace_id": execution.trace_id,
            "span_id": execution.span_id,
            "trajectory_level": trajectory_labels.get("trajectory_level"),
            "messages": _extract_attack_chain_messages(execution.messages, trajectory_labels),
        }
        attack_chain_path = run_dir / TASK_ATTACK_CHAIN_FILENAME
        tmp_attack_chain_path = attack_chain_path.with_suffix(
            f"{attack_chain_path.suffix}.tmp"
        )
        with open(tmp_attack_chain_path, "w") as f:
            json.dump(attack_chain, f, indent=2)
        tmp_attack_chain_path.replace(attack_chain_path)

    def list_incomplete_run_ids(self, task_id: str) -> list[str]:
        """Return run IDs with execution data but no final result for a task/couple."""
        task_dir = self.job_dir / sanitize_for_filename(task_id)
        if not task_dir.exists():
            return []

        run_ids: list[str] = []
        for run_dir in sorted(task_dir.iterdir(), key=lambda path: path.name):
            if not run_dir.is_dir():
                continue
            if (run_dir / TASK_RESULT_FILENAME).exists():
                continue
            if not (run_dir / TASK_EXECUTION_FILENAME).exists():
                continue
            run_ids.append(run_dir.name)
        return run_ids

    def list_resume_run_ids(self, task_id: str) -> list[str]:
        """Return incomplete run IDs expanded by the configured resume repeat count."""
        run_ids = self.list_incomplete_run_ids(task_id)
        if self.resume_partial_repeats <= 1:
            return run_ids
        return [
            run_id
            for run_id in run_ids
            for _ in range(self.resume_partial_repeats)
        ]

    def save_task_run(
        self,
        task: Task[EnvStateT],
        evaluation: EvaluationResult,
        messages: list[ModelMessage],
        usage: RunUsage,
        task_span: LogfireSpan,
        started_at: datetime,
        exception: BaseException | None = None,
        generated_attacks: Mapping[str, InjectionAttack] | None = None,
        attack_score: float | None = None,
        trajectory_level: UptakeLevel | None = None,
        trajectory_labels: TrajectoryLabels | None = None,
        run_id: str | None = None,
    ) -> Path:
        """Save a single task run result and execution data.

        Args:
            task: Task that was executed
            evaluation: Task evaluation results
            messages: Conversation messages
            usage: Token usage
            task_span: Logfire span for trace context
            started_at: Timestamp when task execution started
            exception: Exception if the run failed
            generated_attacks: Generated attack vectors (for attack mode)
            attack_score: Attack score (for attack mode)

        Returns:
            Path to the run directory
        """
        # Generate unique run ID if the run was not created up front for partial persistence.
        if run_id is None:
            run_id = str(uuid.uuid4())[:8]
        run_dir = self.get_task_run_dir(task.id, run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now()
        execution_id = str(uuid.uuid4())[:8]

        # Calculate benign score
        benign_score = (
            sum(evaluation.results.values()) / len(evaluation.results)
            if evaluation.results
            else 0.0
        )

        # Create exception info if needed
        exception_info = ExceptionInfo.from_exception(exception) if exception else None

        # Get trace context
        span_context = task_span.get_span_context() if task_span is not None else None
        trace_id = format(span_context.trace_id, "032x") if span_context else None
        span_id = format(span_context.span_id, "016x") if span_context else None

        attacks_dict: dict[str, Any] | None = None
        if generated_attacks:
            attacks_dict = InjectionAttacksDictTypeAdapter.dump_python(dict(generated_attacks))

        if trajectory_labels is None:
            trajectory_labels = label_trajectory(
                messages,
                attacks=attacks_dict,
                attack_score=attack_score,
            )
        if trajectory_level is None:
            trajectory_level = trajectory_labels.trajectory_level

        # Save result.json (lightweight)
        result = TaskRunResult(
            task_id=task.id,
            run_id=run_id,
            started_at=started_at,
            finished_at=now,
            benign_score=benign_score,
            attack_score=attack_score,
            trajectory_level=trajectory_level,
            attacks=attacks_dict,
            exception_info=exception_info,
        )
        result_path = run_dir / TASK_RESULT_FILENAME
        with open(result_path, "w") as f:
            f.write(result.model_dump_json(indent=2))

        # Save execution.json (heavy)
        execution = TaskRunExecution(
            task_id=task.id,
            run_id=run_id,
            execution_id=execution_id,
            timestamp=now,
            trace_id=trace_id,
            span_id=span_id,
            messages=messages,
            usage=usage,
            attacks=attacks_dict,
            trajectory_labels=trajectory_labels.to_json(),
        )
        self._save_execution_files(
            run_dir,
            execution,
            attacks=attacks_dict,
            trajectory_labels=trajectory_labels.to_json(),
        )

        # Update index
        self._append_to_index(
            task_id=task.id,
            run_id=run_id,
            timestamp=now,
            benign_score=benign_score,
            attack_score=attack_score,
            exception_type=exception_info.exception_type if exception_info else None,
            path=run_dir.relative_to(self.job_dir),
        )

        return run_dir

    def save_couple_run(
        self,
        couple: TaskCouple[EnvStateT],
        benign_eval: EvaluationResult,
        malicious_eval: EvaluationResult,
        messages: list[ModelMessage],
        usage: RunUsage,
        task_span: LogfireSpan,
        started_at: datetime,
        exception: BaseException | None = None,
        generated_attacks: Mapping[str, InjectionAttack] | None = None,
        trajectory_level: UptakeLevel | None = None,
        trajectory_labels: TrajectoryLabels | None = None,
        run_id: str | None = None,
    ) -> Path:
        """Save a task couple run result and execution data.

        Args:
            couple: Task couple that was executed
            benign_eval: Benign task evaluation results
            malicious_eval: Malicious task evaluation results
            messages: Conversation messages
            usage: Token usage
            task_span: Logfire span for trace context
            started_at: Timestamp when task execution started
            exception: Exception if the run failed
            generated_attacks: Generated attack vectors

        Returns:
            Path to the run directory
        """
        # Generate unique run ID if the run was not created up front for partial persistence.
        if run_id is None:
            run_id = str(uuid.uuid4())[:8]
        run_dir = self.get_task_run_dir(couple.id, run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now()
        execution_id = str(uuid.uuid4())[:8]

        # Calculate scores
        benign_score = (
            sum(benign_eval.results.values()) / len(benign_eval.results)
            if benign_eval.results
            else 0.0
        )
        attack_score = (
            sum(malicious_eval.results.values()) / len(malicious_eval.results)
            if malicious_eval.results
            else 0.0
        )

        # Create exception info if needed
        exception_info = ExceptionInfo.from_exception(exception) if exception else None

        # Get trace context
        span_context = task_span.get_span_context() if task_span is not None else None
        trace_id = format(span_context.trace_id, "032x") if span_context else None
        span_id = format(span_context.span_id, "016x") if span_context else None

        attacks_dict: dict[str, Any] | None = None
        if generated_attacks:
            attacks_dict = InjectionAttacksDictTypeAdapter.dump_python(dict(generated_attacks))

        if trajectory_labels is None:
            trajectory_labels = label_trajectory(
                messages,
                attacks=attacks_dict,
                attack_score=attack_score,
            )
        if trajectory_level is None:
            trajectory_level = trajectory_labels.trajectory_level

        # Save result.json (lightweight)
        result = TaskRunResult(
            task_id=couple.id,
            run_id=run_id,
            started_at=started_at,
            finished_at=now,
            benign_score=benign_score,
            attack_score=attack_score,
            trajectory_level=trajectory_level,
            attacks=attacks_dict,
            exception_info=exception_info,
        )
        result_path = run_dir / TASK_RESULT_FILENAME
        with open(result_path, "w") as f:
            f.write(result.model_dump_json(indent=2))

        # Save execution.json (heavy)
        execution = TaskRunExecution(
            task_id=couple.id,
            run_id=run_id,
            execution_id=execution_id,
            timestamp=now,
            trace_id=trace_id,
            span_id=span_id,
            messages=messages,
            usage=usage,
            attacks=attacks_dict,
            trajectory_labels=trajectory_labels.to_json(),
        )
        self._save_execution_files(
            run_dir,
            execution,
            attacks=attacks_dict,
            trajectory_labels=trajectory_labels.to_json(),
        )

        # Update index
        self._append_to_index(
            task_id=couple.id,
            run_id=run_id,
            timestamp=now,
            benign_score=benign_score,
            attack_score=attack_score,
            exception_type=exception_info.exception_type if exception_info else None,
            path=run_dir.relative_to(self.job_dir),
        )
        if attack_score >= 1.0 and exception_info is None:
            self._record_attack_success_case(
                task_id=couple.id,
                attack_score=attack_score,
                run_id=run_id,
            )

        return run_dir

    def _record_attack_success_case(
        self,
        *,
        task_id: str,
        attack_score: float,
        run_id: str,
    ) -> None:
        """Best-effort update of the repository docs table for successful attacks."""
        try:
            record_attack_success_case(
                task_id=task_id,
                model=_success_case_model_name(self.job_config),
                attack_score=attack_score,
                job_name=self.job_config.job_name,
                run_id=run_id,
            )
        except Exception as e:
            logfire.warning(f"Failed to update attack success cases docs: {e}")

    def _append_to_index(
        self,
        task_id: str,
        run_id: str,
        timestamp: datetime,
        benign_score: float | None,
        attack_score: float | None,
        exception_type: str | None,
        path: Path,
    ) -> None:
        """Atomically append to job index.jsonl with file locking."""
        index_path = self.job_dir / INDEX_FILENAME
        lock_path = self.job_dir / f"{INDEX_FILENAME}.lock"

        entry = RunIndexEntry(
            task_id=task_id,
            run_id=run_id,
            timestamp=timestamp,
            benign_score=benign_score,
            attack_score=attack_score,
            exception_type=exception_type,
            path=path,
        )

        with FileLock(lock_path):
            with open(index_path, "a") as f:
                f.write(entry.model_dump_json() + "\n")

    def load_index(self) -> list[RunIndexEntry]:
        """Load all entries from the job index.

        Returns:
            List of index entries
        """
        index_path = self.job_dir / INDEX_FILENAME
        if not index_path.exists():
            return []

        entries = []
        with open(index_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(RunIndexEntry.model_validate_json(line))
        return entries

    def get_run_counts(self) -> dict[str, int]:
        """Count all runs per task from the index.

        Both successful and errored runs count toward the limit.
        Use --retry-on-errors to delete errored runs before resuming.

        Returns:
            Dictionary mapping task_id -> count of runs
        """
        entries = self.load_index()
        return dict(Counter(e.task_id for e in entries))

    def remove_index_entries_by_paths(self, paths_to_remove: set[Path]) -> None:
        """Remove entries from the index by their paths.

        Rewrites the entire index file excluding entries with paths in the given set.

        Args:
            paths_to_remove: Set of paths (relative to job_dir) to remove from index
        """
        index_path = self.job_dir / INDEX_FILENAME
        if not index_path.exists():
            return

        lock_path = self.job_dir / f"{INDEX_FILENAME}.lock"

        with FileLock(lock_path):
            # Load all existing entries
            entries = []
            with open(index_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(RunIndexEntry.model_validate_json(line))

            # Filter out entries with paths in paths_to_remove
            filtered_entries = [entry for entry in entries if entry.path not in paths_to_remove]

            # Rewrite the index file
            with open(index_path, "w") as f:
                for entry in filtered_entries:
                    f.write(entry.model_dump_json() + "\n")


def _return_message_indexes_by_tool_call_id(messages: list[ModelMessage]) -> dict[str, int]:
    indexes: dict[str, int] = {}
    for message_index, message in enumerate(messages):
        if not isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            if isinstance(part, ToolReturnPart | InjectableToolReturnPart):
                indexes[part.tool_call_id] = message_index
    return indexes


def _extract_tool_history_entries(messages: list[ModelMessage]) -> list[ToolHistoryEntry]:
    return_indexes = _return_message_indexes_by_tool_call_id(messages)
    entries: list[ToolHistoryEntry] = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, ModelResponse):
            continue
        entries.extend(
            ToolHistoryEntry(
                message_index=message_index,
                tool_call_id=part.tool_call_id,
                tool_name=part.tool_name,
                args=part.args_as_dict(),
                completed=part.tool_call_id in return_indexes,
                return_message_index=return_indexes.get(part.tool_call_id),
            )
            for part in message.parts
            if isinstance(part, ToolCallPart)
        )
    return entries


def _extract_attack_chain_messages(
    messages: list[ModelMessage],
    trajectory_labels: dict[str, Any],
) -> list[dict[str, Any]]:
    labeled_messages = trajectory_labels.get("messages")
    if not isinstance(labeled_messages, list):
        return []

    attack_chain_messages: list[dict[str, Any]] = []
    for label in labeled_messages:
        if not isinstance(label, dict):
            continue
        message_level = label.get("message_level")
        if not isinstance(message_level, str):
            continue
        if LEVEL_RANK.get(message_level, 0) <= LEVEL_RANK["L0"]:
            continue

        message_index = label.get("message_index")
        if not isinstance(message_index, int):
            continue
        if message_index < 0 or message_index >= len(messages):
            continue

        message = InjectableModelMessagesTypeAdapter.dump_python(
            [messages[message_index]],
            mode="json",
        )[0]
        attack_chain_messages.append(
            {
                "message_index": message_index,
                "message_level": message_level,
                "evidence": label.get("evidence", []),
                "message_kind": message.get("kind"),
                "timestamp": message.get("timestamp"),
                "parts": _extract_message_parts(message),
            }
        )

    return attack_chain_messages


def _extract_message_parts(message: dict[str, Any]) -> list[dict[str, Any]]:
    parts = message.get("parts")
    if not isinstance(parts, list):
        return []

    extracted_parts: list[dict[str, Any]] = []
    for part in parts:
        if not isinstance(part, dict):
            continue

        extracted: dict[str, Any] = {"part_kind": part.get("part_kind")}
        for key in (
            "content",
            "args",
            "tool_name",
            "tool_call_id",
            "outcome",
            "timestamp",
        ):
            if key in part:
                extracted[key] = part[key]
        extracted_parts.append(extracted)

    return extracted_parts


def _success_case_model_name(job_config: JobConfig) -> str:
    agent_config = job_config.agent.config
    if job_config.agent.type == "mini_swe":
        for spec in agent_config.get("config_specs", []):
            spec_path = Path(str(spec)).expanduser()
            if not spec_path.is_file():
                continue
            try:
                spec_config = yaml.safe_load(spec_path.read_text()) or {}
            except Exception:
                continue
            model_config = spec_config.get("model")
            if not isinstance(model_config, dict):
                continue
            raw_model = model_config.get("model_name") or model_config.get("model")
            if raw_model:
                return _display_model_name(str(raw_model))

    raw_model = agent_config.get("model")
    if isinstance(raw_model, dict):
        raw_model = raw_model.get("model_name") or raw_model.get("model")
    if raw_model:
        return _display_model_name(str(raw_model))

    raw_model = agent_config.get("model_name")
    if raw_model:
        return _display_model_name(str(raw_model))

    return job_config.agent.type


def _display_model_name(model_name: str) -> str:
    if model_name.startswith("openai/"):
        return model_name.split("/", 1)[1]
    if model_name.startswith("openai:"):
        return model_name.split(":", 1)[1]
    return model_name


def _save_config_yaml(config_path: Path, job_config: JobConfig) -> None:
    """Save job config to YAML file."""
    config_dict = job_config.model_dump(mode="json")

    header = f"# Job: {job_config.job_name}\n"
    header += f"# Created: {job_config.created_at.isoformat()}\n"
    header += f"# Mode: {job_config.execution_mode}\n\n"

    with open(config_path, "w") as f:
        f.write(header)
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def load_config_yaml(config_path: Path) -> JobConfig:
    """Load job config from YAML file."""
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    return JobConfig.model_validate(config_dict)
