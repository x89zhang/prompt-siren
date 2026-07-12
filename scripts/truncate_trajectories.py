#!/usr/bin/env python3
"""Create partial execution trajectories truncated at selected message points."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Literal

from prompt_siren.job.models import CONFIG_FILENAME, TASK_EXECUTION_METADATA_FILENAME
from prompt_siren.job.naming import sanitize_for_filename
from prompt_siren.trajectory_labeling import label_trajectory, LEVEL_RANK, LEVELS, UptakeLevel

CutSource = Literal["first_occurrence", "first_reach"]
PayloadMatch = Literal["content", "vector_id", "content_or_vector_id"]


def is_probably_execution_file(path: Path) -> bool:
    return path.is_file() and path.name == "execution.json"


def find_execution_paths(paths: list[Path]) -> list[Path]:
    execution_paths: list[Path] = []
    for path in paths:
        if is_probably_execution_file(path):
            execution_paths.append(path)
        elif path.is_dir():
            execution_paths.extend(sorted(path.rglob("execution.json")))
        else:
            raise FileNotFoundError(f"No execution file or directory found at {path}")
    return sorted(dict.fromkeys(execution_paths))


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def load_result_for_execution(execution_path: Path) -> dict[str, Any] | None:
    result_path = execution_path.with_name("result.json")
    if not result_path.exists():
        return None
    return load_json(result_path)


def load_metadata_for_execution(execution_path: Path) -> dict[str, Any] | None:
    metadata_path = execution_path.with_name(TASK_EXECUTION_METADATA_FILENAME)
    if not metadata_path.exists():
        return None
    return load_json(metadata_path)


def find_job_config_path(execution_path: Path) -> Path | None:
    for parent in execution_path.parents:
        config_path = parent / CONFIG_FILENAME
        if config_path.exists():
            return config_path
    return None


def execution_attacks(
    execution: dict[str, Any],
    metadata: dict[str, Any] | None,
) -> Any:
    if isinstance(metadata, dict) and metadata.get("attacks") is not None:
        return metadata.get("attacks")
    return execution.get("attacks")


def ensure_trajectory_labels(
    execution: dict[str, Any],
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    if isinstance(metadata, dict):
        labels = metadata.get("trajectory_labels")
        if isinstance(labels, dict):
            return labels

    labels = execution.get("trajectory_labels")
    if isinstance(labels, dict):
        return labels

    computed = label_trajectory(
        execution.get("messages", []),
        attacks=execution_attacks(execution, metadata),
        attack_score=None,
    )
    return computed.to_json()


def occurrence_index(
    labels: dict[str, Any],
    level: UptakeLevel,
    occurrence: int,
) -> int | None:
    seen = 0
    for label in labels.get("messages", []):
        if not isinstance(label, dict) or label.get("message_level") != level:
            continue
        index = label.get("message_index")
        if not isinstance(index, int):
            continue
        seen += 1
        if seen == occurrence:
            return index
    return None


def target_index(
    labels: dict[str, Any],
    level: UptakeLevel,
    cut_source: CutSource,
    occurrence: int = 1,
) -> int | None:
    if cut_source == "first_occurrence" and occurrence > 1:
        return occurrence_index(labels, level, occurrence)

    source = labels.get(cut_source)
    if isinstance(source, dict):
        point = source.get(level)
        if isinstance(point, dict):
            index = point.get("message_index")
            if isinstance(index, int):
                return index

    if cut_source == "first_occurrence":
        return occurrence_index(labels, level, 1)
    return None


def attack_payload_texts(attacks: Any, match: PayloadMatch) -> list[str]:
    if not isinstance(attacks, dict):
        return []

    needles: list[str] = []
    for vector_id, attack in attacks.items():
        if match in ("vector_id", "content_or_vector_id") and isinstance(vector_id, str):
            needles.append(vector_id)
        if match not in ("content", "content_or_vector_id") or not isinstance(attack, dict):
            continue
        content = attack.get("content")
        if not isinstance(content, str) or not content:
            continue
        needles.append(content)
        needles.extend(
            line.strip()
            for line in content.splitlines()
            if len(line.strip()) >= 32
        )

    return sorted(dict.fromkeys(needles), key=len, reverse=True)


def first_payload_exposure_index(
    messages: list[Any],
    attacks: Any,
    *,
    match: PayloadMatch,
) -> int | None:
    needles = attack_payload_texts(attacks, match)
    if not needles:
        return None

    for index, message in enumerate(messages):
        serialized = json.dumps(message, ensure_ascii=False)
        if any(needle in serialized for needle in needles):
            return index
    return None


def kept_message_count(index: int, include_trigger: bool) -> int:
    return index + 1 if include_trigger else index


def level_max(levels: list[UptakeLevel]) -> UptakeLevel:
    return max(levels, key=lambda item: LEVEL_RANK[item], default="L0")


def recompute_first_occurrence(
    message_labels: list[dict[str, Any]],
) -> dict[UptakeLevel, dict[str, Any] | None]:
    first: dict[UptakeLevel, dict[str, Any] | None] = dict.fromkeys(LEVELS[1:], None)
    for label in message_labels:
        level = label.get("message_level")
        if level not in LEVEL_RANK or level == "L0" or first[level] is not None:
            continue
        first[level] = {
            "message_index": label.get("message_index"),
            "evidence": label.get("evidence", []),
        }
    return first


def recompute_first_reach(
    message_labels: list[dict[str, Any]],
) -> dict[UptakeLevel, dict[str, Any] | None]:
    first: dict[UptakeLevel, dict[str, Any] | None] = dict.fromkeys(LEVELS[1:], None)
    current: UptakeLevel = "L0"
    for label in message_labels:
        level = label.get("message_level")
        if level not in LEVEL_RANK or LEVEL_RANK[level] <= LEVEL_RANK[current]:
            continue
        current = level
        for threshold in LEVELS[1 : LEVEL_RANK[current] + 1]:
            if first[threshold] is None:
                first[threshold] = {
                    "message_index": label.get("message_index"),
                    "reached_by": level,
                    "evidence": label.get("evidence", []),
                }
    return first


def truncate_labels(labels: dict[str, Any], keep_count: int) -> dict[str, Any]:
    message_labels = [
        label
        for label in labels.get("messages", [])
        if isinstance(label, dict)
        and isinstance(label.get("message_index"), int)
        and label["message_index"] < keep_count
    ]
    levels = [
        label["message_level"]
        for label in message_labels
        if label.get("message_level") in LEVEL_RANK
    ]
    return {
        "trajectory_level": level_max(levels),
        "first_occurrence": recompute_first_occurrence(message_labels),
        "first_reach": recompute_first_reach(message_labels),
        "messages": message_labels,
    }


def truncate_execution(
    execution: dict[str, Any],
    *,
    keep_count: int,
) -> dict[str, Any]:
    truncated = copy.deepcopy(execution)
    truncated["messages"] = truncated.get("messages", [])[:keep_count]
    truncated["resume_state"] = None
    for metadata_key in ("attacks", "trajectory_labels", "truncation"):
        truncated.pop(metadata_key, None)
    return truncated


def truncate_result(
    result: dict[str, Any] | None,
    *,
    execution: dict[str, Any],
    trajectory_level: UptakeLevel,
    keep_count: int,
    source_execution_path: Path,
    truncation: dict[str, Any],
) -> dict[str, Any]:
    truncated = copy.deepcopy(result) if result is not None else {}
    truncated["task_id"] = truncated.get("task_id") or execution.get("task_id")
    truncated["run_id"] = truncated.get("run_id") or execution.get("run_id")
    truncated["trajectory_level"] = trajectory_level
    truncated["attack_score"] = None
    truncated["benign_score"] = None
    truncated["exception_info"] = None
    truncated["truncation"] = truncation | {
        "source_execution_path": str(source_execution_path),
        "kept_message_count": keep_count,
    }
    return truncated


def common_root(paths: list[Path]) -> Path:
    if len(paths) == 1:
        return paths[0].parent
    return Path(os.path.commonpath([path.parent for path in paths]))


def output_run_dir(
    *,
    output_dir: Path,
    root: Path,
    execution_path: Path,
    suffix: str,
) -> Path:
    relative_parent = execution_path.parent.relative_to(root)
    return output_dir / relative_parent / f"{execution_path.parent.name}__truncated_{suffix}"


def resume_ready_run_dir(
    *,
    resume_job_dir: Path,
    execution: dict[str, Any],
    execution_path: Path,
    suffix: str,
) -> Path:
    task_id = execution.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError(f"Cannot build resume-ready path without task_id in {execution_path}")
    source_run_id = execution_path.parent.name
    run_id = f"{source_run_id}__truncated_{suffix}"
    return resume_job_dir / sanitize_for_filename(task_id) / run_id


def ensure_resume_job_config(resume_job_dir: Path, execution_path: Path) -> Path:
    source_config = find_job_config_path(execution_path)
    if source_config is None:
        raise FileNotFoundError(f"Could not find source {CONFIG_FILENAME} for {execution_path}")

    target_config = resume_job_dir / CONFIG_FILENAME
    target_config.parent.mkdir(parents=True, exist_ok=True)
    if not target_config.exists():
        shutil.copy2(source_config, target_config)
    return target_config


def write_truncated_record(
    *,
    record: dict[str, Any],
    truncated_execution: dict[str, Any],
    truncated_result: dict[str, Any],
    output_dir: Path | None,
    resume_job_dir: Path | None,
    root: Path,
    execution_path: Path,
    suffix: str,
) -> dict[str, Any]:
    if output_dir is not None:
        run_dir = output_run_dir(
            output_dir=output_dir,
            root=root,
            execution_path=execution_path,
            suffix=suffix,
        )
        dump_json(run_dir / "execution.json", truncated_execution)
        dump_json(run_dir / "result.json", truncated_result)
        record["status"] = "written"
        record["output_execution_path"] = str(run_dir / "execution.json")
        record["output_result_path"] = str(run_dir / "result.json")

    if resume_job_dir is not None:
        ensure_resume_job_config(resume_job_dir, execution_path)
        run_dir = resume_ready_run_dir(
            resume_job_dir=resume_job_dir,
            execution=truncated_execution,
            execution_path=execution_path,
            suffix=suffix,
        )
        truncated_execution["run_id"] = run_dir.name
        dump_json(run_dir / "execution.json", truncated_execution)
        record["status"] = "written_resume_ready"
        record["resume_job_dir"] = str(resume_job_dir)
        record["resume_execution_path"] = str(run_dir / "execution.json")

    return record


def truncate_one(
    execution_path: Path,
    *,
    levels: list[UptakeLevel],
    cut_source: CutSource,
    occurrence: int,
    include_trigger: bool,
    keep_message_counts: list[int],
    payload_offsets: list[int],
    payload_match: PayloadMatch,
    output_dir: Path | None,
    resume_job_dir: Path | None,
    root: Path,
) -> list[dict[str, Any]]:
    execution = load_json(execution_path)
    metadata = load_metadata_for_execution(execution_path)
    result = load_result_for_execution(execution_path)
    labels = ensure_trajectory_labels(execution, metadata)
    attacks = execution_attacks(execution, metadata)
    messages = execution.get("messages", [])
    records = []

    for level in levels:
        index = target_index(labels, level, cut_source, occurrence)
        if index is None:
            records.append(
                {
                    "source_execution_path": str(execution_path),
                    "method": "label",
                    "target_level": level,
                    "cut_source": cut_source,
                    "occurrence": occurrence,
                    "status": "missing_level_occurrence" if occurrence > 1 else "missing_level",
                }
            )
            continue

        keep_count = kept_message_count(index, include_trigger)
        if keep_count < 0 or keep_count > len(messages):
            raise ValueError(
                f"Invalid truncation count {keep_count} for {execution_path} at {level}"
            )

        truncation = {
            "method": "label",
            "target_level": level,
            "cut_source": cut_source,
            "include_trigger_message": include_trigger,
            "occurrence": occurrence,
        }
        suffix = f"{cut_source}_{level}"
        if occurrence > 1:
            suffix = f"{suffix}_occurrence_{occurrence}"
        truncated_labels = truncate_labels(labels, keep_count)
        truncated_execution = truncate_execution(
            execution,
            keep_count=keep_count,
        )
        truncated_result = truncate_result(
            result,
            execution=truncated_execution,
            trajectory_level=truncated_labels["trajectory_level"],
            keep_count=keep_count,
            source_execution_path=execution_path,
            truncation=truncation,
        )

        record = {
            "source_execution_path": str(execution_path),
            "method": "label",
            "target_level": level,
            "cut_source": cut_source,
            "occurrence": occurrence,
            "trigger_message_index": index,
            "kept_message_count": keep_count,
            "original_message_count": len(messages),
            "truncated_trajectory_level": truncated_labels["trajectory_level"],
            "status": "planned",
        }
        records.append(
            write_truncated_record(
                record=record,
                truncated_execution=truncated_execution,
                truncated_result=truncated_result,
                output_dir=output_dir,
                resume_job_dir=resume_job_dir,
                root=root,
                execution_path=execution_path,
                suffix=suffix,
            )
        )

    for keep_count in keep_message_counts:
        if keep_count < 0 or keep_count > len(messages):
            records.append(
                {
                    "source_execution_path": str(execution_path),
                    "method": "message_count",
                    "keep_message_count": keep_count,
                    "original_message_count": len(messages),
                    "status": "count_out_of_range",
                }
            )
            continue

        truncation = {
            "method": "message_count",
            "keep_message_count": keep_count,
        }
        suffix = f"message_count_{keep_count}"
        truncated_labels = truncate_labels(labels, keep_count)
        truncated_execution = truncate_execution(
            execution,
            keep_count=keep_count,
        )
        truncated_result = truncate_result(
            result,
            execution=truncated_execution,
            trajectory_level=truncated_labels["trajectory_level"],
            keep_count=keep_count,
            source_execution_path=execution_path,
            truncation=truncation,
        )

        record = {
            "source_execution_path": str(execution_path),
            "method": "message_count",
            "keep_message_count": keep_count,
            "kept_message_count": keep_count,
            "original_message_count": len(messages),
            "truncated_trajectory_level": truncated_labels["trajectory_level"],
            "status": "planned",
        }
        records.append(
            write_truncated_record(
                record=record,
                truncated_execution=truncated_execution,
                truncated_result=truncated_result,
                output_dir=output_dir,
                resume_job_dir=resume_job_dir,
                root=root,
                execution_path=execution_path,
                suffix=suffix,
            )
        )

    exposure_index = first_payload_exposure_index(
        messages,
        attacks,
        match=payload_match,
    )
    for offset in payload_offsets:
        if exposure_index is None:
            records.append(
                {
                    "source_execution_path": str(execution_path),
                    "method": "payload_exposure",
                    "payload_offset": offset,
                    "payload_match": payload_match,
                    "status": "missing_payload_exposure",
                }
            )
            continue

        target_message_index = exposure_index + offset
        keep_count = target_message_index + 1
        if keep_count < 0 or keep_count > len(messages):
            records.append(
                {
                    "source_execution_path": str(execution_path),
                    "method": "payload_exposure",
                    "payload_offset": offset,
                    "payload_match": payload_match,
                    "payload_exposure_message_index": exposure_index,
                    "target_message_index": target_message_index,
                    "original_message_count": len(messages),
                    "status": "offset_out_of_range",
                }
            )
            continue

        truncation = {
            "method": "payload_exposure",
            "payload_match": payload_match,
            "payload_offset": offset,
            "payload_exposure_message_index": exposure_index,
            "target_message_index": target_message_index,
        }
        suffix = f"payload_exposure_after_{offset}"
        truncated_labels = truncate_labels(labels, keep_count)
        truncated_execution = truncate_execution(
            execution,
            keep_count=keep_count,
        )
        truncated_result = truncate_result(
            result,
            execution=truncated_execution,
            trajectory_level=truncated_labels["trajectory_level"],
            keep_count=keep_count,
            source_execution_path=execution_path,
            truncation=truncation,
        )

        record = {
            "source_execution_path": str(execution_path),
            "method": "payload_exposure",
            "payload_offset": offset,
            "payload_match": payload_match,
            "payload_exposure_message_index": exposure_index,
            "target_message_index": target_message_index,
            "kept_message_count": keep_count,
            "original_message_count": len(messages),
            "truncated_trajectory_level": truncated_labels["trajectory_level"],
            "status": "planned",
        }
        records.append(
            write_truncated_record(
                record=record,
                truncated_execution=truncated_execution,
                truncated_result=truncated_result,
                output_dir=output_dir,
                resume_job_dir=resume_job_dir,
                root=root,
                execution_path=execution_path,
                suffix=suffix,
            )
        )

    return records


def parse_level(value: str) -> UptakeLevel:
    normalized = value.upper()
    if normalized not in LEVEL_RANK or normalized == "L0":
        raise argparse.ArgumentTypeError("level must be one of L1, L2, L3, L4, L5")
    return normalized  # type: ignore[return-value]


def parse_occurrence(value: str) -> int:
    try:
        occurrence = int(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError("occurrence must be an integer") from e
    if occurrence < 1:
        raise argparse.ArgumentTypeError("occurrence must be at least 1")
    return occurrence


def parse_payload_offset(value: str) -> int:
    try:
        offset = int(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError("payload offset must be an integer") from e
    if offset < 0:
        raise argparse.ArgumentTypeError("payload offset must be non-negative")
    return offset


def parse_keep_message_count(value: str) -> int:
    try:
        count = int(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError("message count must be an integer") from e
    if count < 0:
        raise argparse.ArgumentTypeError("message count must be non-negative")
    return count


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Truncate Prompt Siren execution trajectories at selected message points."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="One or more execution.json files or directories to scan recursively.",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        type=parse_level,
        default=["L2", "L3", "L4", "L5"],
        help="Target levels to truncate at. Defaults to L2 L3 L4 L5.",
    )
    parser.add_argument(
        "--cut-source",
        choices=("first_occurrence", "first_reach"),
        default="first_occurrence",
        help="Which label timestamp to use as the truncation point.",
    )
    parser.add_argument(
        "--before-trigger",
        action="store_true",
        help="Drop the trigger message instead of truncating immediately after it.",
    )
    parser.add_argument(
        "--occurrence",
        type=parse_occurrence,
        default=1,
        help=(
            "Which exact level occurrence to truncate at when --cut-source is "
            "first_occurrence. Defaults to 1."
        ),
    )
    parser.add_argument(
        "--skip-labels",
        action="store_true",
        help="Do not create the default label-based truncations.",
    )
    parser.add_argument(
        "--keep-message-count",
        nargs="+",
        type=parse_keep_message_count,
        default=[],
        help=(
            "Create truncations that keep exactly the first N messages. "
            "Can be provided with one or more counts, for example: "
            "--keep-message-count 20 40."
        ),
    )
    parser.add_argument(
        "--payload-offsets",
        nargs="+",
        type=parse_payload_offset,
        default=[],
        help=(
            "Create truncations at the first payload exposure plus each offset. "
            "Offset 0 keeps through the exposure message; offset 1 keeps one later message."
        ),
    )
    parser.add_argument(
        "--payload-match",
        choices=("content", "vector_id", "content_or_vector_id"),
        default="content",
        help="What to search for when finding the first payload exposure.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where truncated execution/result files are written. If omitted, only JSONL plans are printed.",
    )
    parser.add_argument(
        "--resume-job-dir",
        type=Path,
        default=None,
        help=(
            "Write truncated executions as incomplete runs under a resumable job directory. "
            "This copies the source config.yaml and intentionally does not write result.json."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.occurrence != 1 and args.cut_source != "first_occurrence":
        raise SystemExit("--occurrence can only be used with --cut-source first_occurrence")
    execution_paths = find_execution_paths(args.paths)
    root = common_root(execution_paths)
    levels = [] if args.skip_labels else args.levels
    records: list[dict[str, Any]] = []
    for execution_path in execution_paths:
        records.extend(
            truncate_one(
                execution_path,
                levels=levels,
                cut_source=args.cut_source,
                occurrence=args.occurrence,
                include_trigger=not args.before_trigger,
                keep_message_counts=args.keep_message_count,
                payload_offsets=args.payload_offsets,
                payload_match=args.payload_match,
                output_dir=args.output_dir,
                resume_job_dir=args.resume_job_dir,
                root=root,
            )
        )

    for record in records:
        print(json.dumps(record, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
