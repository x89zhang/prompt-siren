# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Persistence layer for job-based task execution results."""

from __future__ import annotations

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
from pydantic_ai.messages import ModelMessage
from pydantic_ai.usage import RunUsage

from ..attack_success_cases import record_attack_success_case
from ..tasks import EvaluationResult, Task, TaskCouple
from ..trajectory_labeling import label_trajectory, TrajectoryLabels, UptakeLevel
from ..types import InjectionAttack, InjectionAttacksDictTypeAdapter
from .models import (
    CONFIG_FILENAME,
    ExceptionInfo,
    INDEX_FILENAME,
    JobConfig,
    RunIndexEntry,
    TASK_EXECUTION_FILENAME,
    TASK_RESULT_FILENAME,
    TaskRunExecution,
    TaskRunResult,
    TaskRunResumeState,
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
                    execution.json   # Full execution data
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
        execution_path = run_dir / TASK_EXECUTION_FILENAME
        tmp_path = execution_path.with_suffix(f"{execution_path.suffix}.tmp")
        with open(tmp_path, "w") as f:
            f.write(execution.model_dump_json(indent=2))
        tmp_path.replace(execution_path)
        return run_dir

    def load_execution(self, task_id: str, run_id: str) -> TaskRunExecution:
        """Load execution data for a task run."""
        execution_path = self.get_task_run_dir(task_id, run_id) / TASK_EXECUTION_FILENAME
        return TaskRunExecution.model_validate_json(execution_path.read_text())

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
        execution_path = run_dir / TASK_EXECUTION_FILENAME
        with open(execution_path, "w") as f:
            f.write(execution.model_dump_json(indent=2))

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
        execution_path = run_dir / TASK_EXECUTION_FILENAME
        with open(execution_path, "w") as f:
            f.write(execution.model_dump_json(indent=2))

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
