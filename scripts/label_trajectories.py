#!/usr/bin/env python3
"""Label Prompt Siren execution trajectories with message-level uptake levels."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import sys
from pathlib import Path
from typing import Any

import yaml
from prompt_siren.providers import infer_model
from prompt_siren.trajectory_labeling import (
    JudgeAuditSettings,
    label_trajectory,
    label_trajectory_async,
    label_trajectory_with_judge_audit,
    messages_to_dicts,
    serializable_attacks,
)


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


def load_result_for_execution(execution_path: Path) -> dict[str, Any] | None:
    result_path = execution_path.with_name("result.json")
    if not result_path.exists():
        return None
    return load_json(result_path)


def find_job_config(execution_path: Path) -> dict[str, Any] | None:
    for parent in execution_path.parents:
        config_path = parent / "config.yaml"
        if config_path.exists():
            with config_path.open(encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle)
            return loaded if isinstance(loaded, dict) else None
    return None


def agent_model_name_from_job_config(config: dict[str, Any] | None) -> str | None:
    if not config:
        return None
    agent = config.get("agent")
    if not isinstance(agent, dict):
        return None
    agent_config = agent.get("config")
    if not isinstance(agent_config, dict):
        return None

    model = agent_config.get("model")
    if isinstance(model, str):
        return model

    config_specs = agent_config.get("config_specs")
    if isinstance(config_specs, list):
        for spec in config_specs:
            if isinstance(spec, str) and spec.startswith("model.model_name="):
                return spec.split("=", 1)[1]
    return None


def part_summary(part: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {"part_kind": part.get("part_kind")}
    if part.get("tool_name") is not None:
        summary["tool_name"] = part.get("tool_name")
    if part.get("tool_call_id") is not None:
        summary["tool_call_id"] = part.get("tool_call_id")
    if part.get("outcome") is not None:
        summary["outcome"] = part.get("outcome")
    return summary


def labels_to_execution(execution: dict[str, Any], labels: Any) -> dict[str, Any]:
    relabeled = copy.deepcopy(execution)
    relabeled["trajectory_labels"] = labels.to_json()
    if "trajectory_level" in relabeled:
        relabeled["trajectory_level"] = labels.trajectory_level
    return relabeled


def labels_to_record(
    *,
    execution_path: Path,
    execution: dict[str, Any],
    result: dict[str, Any] | None,
    labels: Any,
) -> dict[str, Any]:
    message_dicts = messages_to_dicts(execution.get("messages", []))
    result_path = execution_path.with_name("result.json")
    return {
        "schema_version": "message-uptake-labels-v1",
        "source_execution_path": str(execution_path),
        "source_result_path": str(result_path) if result_path.exists() else None,
        "task_id": execution.get("task_id"),
        "run_id": execution.get("run_id"),
        "execution_id": execution.get("execution_id"),
        "attack_score": result.get("attack_score") if result else None,
        "benign_score": result.get("benign_score") if result else None,
        "trajectory_level": labels.trajectory_level,
        "first_occurrence": labels.first_occurrence,
        "first_reach": labels.first_reach,
        "messages": [
            {
                "message_index": label.message_index,
                "kind": message_dicts[label.message_index].get("kind"),
                "timestamp": message_dicts[label.message_index].get("timestamp"),
                "message_level": label.message_level,
                "parts": [
                    part_summary(part)
                    for part in message_dicts[label.message_index].get("parts", [])
                    if isinstance(part, dict)
                ],
                "evidence": label.evidence,
            }
            for label in labels.messages
        ],
    }


async def label_execution(
    execution_path: Path,
    *,
    l2_judge_model_name: str | None,
    l2_threshold: float,
    use_job_agent_model: bool,
    message_judge_model_name: str | None,
    use_job_agent_model_for_message_judge: bool,
    judge_audit_max_attempts: int,
    output_format: str,
) -> dict[str, Any]:
    execution = load_json(execution_path)
    result = load_result_for_execution(execution_path)
    attacks = serializable_attacks(execution.get("attacks"))
    attack_score = result.get("attack_score") if result else None

    config_model_name = agent_model_name_from_job_config(find_job_config(execution_path))
    message_model_name = message_judge_model_name
    if message_model_name is None and use_job_agent_model_for_message_judge:
        message_model_name = config_model_name

    if message_model_name:
        message_judge_model = infer_model(message_model_name)
        labels = await label_trajectory_with_judge_audit(
            execution.get("messages", []),
            attacks=attacks,
            attack_score=attack_score,
            message_judge_model=message_judge_model,
            judge_audit_settings=JudgeAuditSettings(max_attempts=judge_audit_max_attempts),
        )
    else:
        model_name = l2_judge_model_name
        if model_name is None and use_job_agent_model:
            model_name = config_model_name

        if model_name:
            judge_model = infer_model(model_name)
            labels = await label_trajectory_async(
                execution.get("messages", []),
                attacks=attacks,
                attack_score=attack_score,
                l2_judge_model=judge_model,
                l2_threshold=l2_threshold,
                l3_pattern_model=judge_model,
                task_id=execution.get("task_id"),
            )
        else:
            labels = label_trajectory(
                execution.get("messages", []),
                attacks=attacks,
                attack_score=attack_score,
            )

    if output_format == "execution":
        return labels_to_execution(execution, labels)

    return labels_to_record(
        execution_path=execution_path,
        execution=execution,
        result=result,
        labels=labels,
    )


def write_records(records: list[dict[str, Any]], output_path: Path | None, pretty: bool) -> None:
    if output_path is None:
        for record in records:
            print(json.dumps(record, ensure_ascii=False, sort_keys=True))
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        if pretty:
            json.dump(records, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            return

        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def write_jsonl_record(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
    handle.write("\n")
    handle.flush()


def write_pretty_json(record: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def mirrored_execution_output_path(output_root: Path, execution_path: Path) -> Path:
    try:
        relative = execution_path.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        relative = Path(execution_path.name)
    return output_root / relative


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label Prompt Siren execution trajectories with local message uptake levels."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="One or more execution.json files or directories to scan recursively.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path. Defaults to JSONL on stdout.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Write one pretty JSON array instead of JSONL. Only applies with --output.",
    )
    parser.add_argument(
        "--l2-judge-model",
        default=None,
        help="Optional model name for L3 verbal-uptake LLM judge. If omitted, no verbal-uptake judge is used unless --use-job-agent-model is set.",
    )
    parser.add_argument(
        "--use-job-agent-model",
        action="store_true",
        help="Use the agent model recorded in the job config for L3 verbal-uptake judging.",
    )
    parser.add_argument(
        "--l2-threshold",
        type=float,
        default=7.0,
        help="Minimum 1-10 LLM judge score required to label an agent message as L3.",
    )
    parser.add_argument(
        "--message-judge-model",
        default=None,
        help=(
            "Optional model name for full L0-L5 message judging. When set, the "
            "LLM proposes each message label and deterministic rules audit it."
        ),
    )
    parser.add_argument(
        "--use-job-agent-model-for-message-judge",
        action="store_true",
        help="Use the job config agent model for full L0-L5 message judging.",
    )
    parser.add_argument(
        "--judge-audit-max-attempts",
        type=int,
        default=3,
        help="Maximum LLM judge retries per message after deterministic audit failures.",
    )
    parser.add_argument(
        "--output-format",
        choices=("labels", "execution"),
        default="labels",
        help=(
            "Output labels summary records, or full execution-shaped records with "
            "trajectory_labels and trajectory_level replaced."
        ),
    )
    return parser.parse_args(argv)


async def async_main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    execution_paths = find_execution_paths(args.paths)
    total = len(execution_paths)

    if args.output_format == "execution" and args.output is not None:
        write_single_json = total == 1 and args.output.suffix == ".json"
        for index, path in enumerate(execution_paths, start=1):
            print(f"[{index}/{total}] labeling {path}", file=sys.stderr, flush=True)
            record = await label_execution(
                path,
                l2_judge_model_name=args.l2_judge_model,
                l2_threshold=args.l2_threshold,
                use_job_agent_model=args.use_job_agent_model,
                message_judge_model_name=args.message_judge_model,
                use_job_agent_model_for_message_judge=args.use_job_agent_model_for_message_judge,
                judge_audit_max_attempts=args.judge_audit_max_attempts,
                output_format=args.output_format,
            )
            output_path = args.output if write_single_json else mirrored_execution_output_path(args.output, path)
            write_pretty_json(record, output_path)
            print(f"[{index}/{total}] wrote {output_path}", file=sys.stderr, flush=True)
        return 0

    if args.output is not None and not args.pretty:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            for index, path in enumerate(execution_paths, start=1):
                print(f"[{index}/{total}] labeling {path}", file=sys.stderr, flush=True)
                record = await label_execution(
                    path,
                    l2_judge_model_name=args.l2_judge_model,
                    l2_threshold=args.l2_threshold,
                    use_job_agent_model=args.use_job_agent_model,
                    message_judge_model_name=args.message_judge_model,
                    use_job_agent_model_for_message_judge=args.use_job_agent_model_for_message_judge,
                    judge_audit_max_attempts=args.judge_audit_max_attempts,
                    output_format=args.output_format,
                )
                write_jsonl_record(handle, record)
                print(f"[{index}/{total}] wrote {args.output}", file=sys.stderr, flush=True)
        return 0

    records = [
        await label_execution(
            path,
            l2_judge_model_name=args.l2_judge_model,
            l2_threshold=args.l2_threshold,
            use_job_agent_model=args.use_job_agent_model,
            message_judge_model_name=args.message_judge_model,
            use_job_agent_model_for_message_judge=args.use_job_agent_model_for_message_judge,
            judge_audit_max_attempts=args.judge_audit_max_attempts,
            output_format=args.output_format,
        )
        for path in execution_paths
    ]
    write_records(records, args.output, args.pretty)
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
