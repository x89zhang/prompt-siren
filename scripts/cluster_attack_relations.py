#!/usr/bin/env python3
"""Discover provisional attack-conditioned topics from unlabeled analysis units."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

ANALYSIS_FILENAME = "attack_analysis_units.json"
DocumentView = Literal["raw", "normalized"]
STRUCTURAL_STOP_WORDS = {
    "abs_path",
    "abspath",
    "action",
    "agent_message",
    "assistant",
    "attack",
    "context",
    "hex_id",
    "next",
    "observation",
    "observation_to_agent_transition",
    "result_observed",
    "thought",
    "timestamp",
    "tool",
    "tool_call_id",
    "tool_outcome",
    "tool_succeeded",
    "unit",
    "unknown",
    "url",
    "uuid",
}


def _group_name(unit: dict[str, Any], *, group_by_tool: bool) -> str:
    unit_type = str(unit.get("unit_type", "unknown"))
    if not group_by_tool:
        return unit_type
    tool_name = unit.get("tool_name")
    return f"{unit_type}:{tool_name or 'no-tool'}"


def _behavior_document(conditioned_document: str) -> str:
    """Remove the repeated attack prefix for c-TF-IDF topic representation."""
    marker = "[UNIT="
    marker_index = conditioned_document.find(marker)
    if marker_index < 0:
        return conditioned_document
    return conditioned_document[marker_index:].strip()


def _attack_document(payload: dict[str, Any]) -> str | None:
    """Extract payload text without copying it into persisted assignments."""
    sections: list[str] = []
    for item in payload.get("attack_context", []):
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if isinstance(content, str) and content.strip():
            sections.append(content.strip())
    return "\n\n---\n\n".join(sections) or None


def collect_documents(
    job_dir: Path,
    *,
    document_view: DocumentView,
    group_by_tool: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Collect every unlabeled unit; no relation-based filtering is applied."""
    document_key = f"{document_view}_document"
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(job_dir.rglob(ANALYSIS_FILENAME)):
        payload = json.loads(path.read_text())
        attack_document = _attack_document(payload)
        for unit in payload.get("units", []):
            if not isinstance(unit, dict):
                continue
            document = unit.get(document_key)
            if not isinstance(document, str) or not document.strip():
                continue
            message_indices = unit.get("message_indices", [])
            from_message_id = unit.get("from_message_id")
            to_message_id = unit.get("to_message_id")
            if unit.get("unit_type") == "observation_to_agent_transition":
                if from_message_id is None and message_indices:
                    from_message_id = f"m{message_indices[0]}"
                if to_message_id is None and len(message_indices) > 1:
                    to_message_id = f"m{message_indices[-1]}"
            groups[_group_name(unit, group_by_tool=group_by_tool)].append(
                {
                    "embedding_document": document,
                    "topic_document": _behavior_document(document),
                    "attack_document": attack_document,
                    "unit_id": unit.get("unit_id"),
                    "unit_type": unit.get("unit_type"),
                    "trajectory_id": unit.get("trajectory_id"),
                    "task_id": payload.get("task_id"),
                    "run_id": payload.get("run_id"),
                    "group_id": unit.get("group_id"),
                    "source_event_ids": unit.get("source_event_ids", []),
                    "message_indices": message_indices,
                    "from_message_id": from_message_id,
                    "to_message_id": to_message_id,
                    "tool_name": unit.get("tool_name"),
                    "tool_call_id": unit.get("tool_call_id"),
                    "distance": unit.get("distance"),
                    "observation_event_id": unit.get("observation_event_id"),
                    "next_assistant_event_id": unit.get("next_assistant_event_id"),
                    "attack_context_status": unit.get("attack_context_status"),
                    "source_file": str(path.relative_to(job_dir)),
                }
            )
    return dict(groups)


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _membership_probability(probabilities: Any, index: int) -> float | None:
    if probabilities is None:
        return None
    row = probabilities[index]
    if hasattr(row, "max"):
        return float(row.max())
    if isinstance(row, int | float):
        return float(row)
    return None


def _sample_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key
        in {
            "unit_id",
            "unit_type",
            "trajectory_id",
            "task_id",
            "run_id",
            "group_id",
            "source_event_ids",
            "message_indices",
            "from_message_id",
            "to_message_id",
            "tool_name",
            "tool_call_id",
            "observation_event_id",
            "next_assistant_event_id",
            "source_file",
            "topic_id",
            "topic_namespace",
            "topic_membership_probability",
            "attack_similarity",
        }
    }


def _compact_topic_info(topic_info: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop BERTopic's copied representative documents from persisted output."""
    return [
        {key: value for key, value in topic.items() if key != "Representative_Docs"}
        for topic in topic_info
    ]


def _cosine_similarity(left: Any, right: Any) -> float:
    import numpy as np

    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    denominator = np.linalg.norm(left_array) * np.linalg.norm(right_array)
    if denominator == 0:
        return 0.0
    return float(np.dot(left_array, right_array) / denominator)


def _attack_similarity_summary(values: list[float]) -> dict[str, Any]:
    import numpy as np

    array = np.asarray(values, dtype=float)
    top_values = np.sort(array)[-min(5, len(array)) :]
    return {
        "unit_count": len(values),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "max": float(np.max(array)),
        "top5_mean": float(np.mean(top_values)),
    }


def _rank_topics_by_attack_similarity(
    topic_info: list[dict[str, Any]],
    assignments: list[dict[str, Any]],
    *,
    group_name: str,
) -> list[dict[str, Any]]:
    values_by_topic: dict[int, list[float]] = defaultdict(list)
    for assignment in assignments:
        similarity = assignment.get("attack_similarity")
        if isinstance(similarity, int | float):
            values_by_topic[int(assignment["topic_id"])].append(float(similarity))

    info_by_topic = {int(topic["Topic"]): topic for topic in topic_info}
    for topic_id, values in values_by_topic.items():
        if topic_id in info_by_topic:
            info_by_topic[topic_id]["attack_similarity"] = _attack_similarity_summary(
                values
            )

    ranked = sorted(
        (
            topic
            for topic in topic_info
            if int(topic.get("Topic", -1)) != -1 and "attack_similarity" in topic
        ),
        key=lambda topic: topic["attack_similarity"]["p90"],
        reverse=True,
    )
    candidates: list[dict[str, Any]] = []
    for rank, topic in enumerate(ranked, start=1):
        topic["attack_proximity_rank"] = rank
        candidates.append(
            {
                "rank": rank,
                "topic_id": int(topic["Topic"]),
                "topic_namespace": f"{group_name}/topic_{int(topic['Topic'])}",
                "name": topic.get("Name"),
                "count": topic.get("Count"),
                "ranking_metric": "p90",
                "ranking_score": topic["attack_similarity"]["p90"],
                "attack_similarity": topic["attack_similarity"],
            }
        )
    return candidates


def build_review_samples(
    assignments: list[dict[str, Any]],
    *,
    samples_per_bucket: int,
    random_seed: int,
) -> dict[str, Any]:
    """Export representative, boundary, random, and outlier units for human review."""
    rng = random.Random(random_seed)
    by_topic: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for assignment in assignments:
        by_topic[int(assignment["topic_id"])].append(assignment)

    topics: dict[str, Any] = {}
    for topic_id, records in sorted(by_topic.items()):
        if topic_id == -1:
            continue
        ranked = sorted(
            records,
            key=lambda record: record.get("topic_membership_probability") or 0.0,
            reverse=True,
        )
        random_records = rng.sample(records, min(samples_per_bucket, len(records)))
        topics[str(topic_id)] = {
            "representative": [
                _sample_record(record) for record in ranked[:samples_per_bucket]
            ],
            "boundary": [
                _sample_record(record) for record in ranked[-samples_per_bucket:]
            ],
            "random": [_sample_record(record) for record in random_records],
        }

    outliers = by_topic.get(-1, [])
    return {
        "topics": topics,
        "outliers": [
            _sample_record(record)
            for record in rng.sample(outliers, min(samples_per_bucket, len(outliers)))
        ],
    }


def fit_topics(
    groups: dict[str, list[dict[str, Any]]],
    *,
    document_view: DocumentView,
    min_documents: int,
    min_topic_size: int,
    model_dir: Path | None,
    samples_per_bucket: int,
    random_seed: int,
    embedding_model_name: str,
) -> dict[str, Any]:
    try:
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer
        from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
    except ImportError as exc:
        raise SystemExit(
            "BERTopic is not installed. Install it with `uv sync --extra topics`."
        ) from exc

    embedding_model = SentenceTransformer(embedding_model_name)
    topic_stop_words = sorted(set(ENGLISH_STOP_WORDS) | STRUCTURAL_STOP_WORDS)

    output: dict[str, Any] = {"groups": {}}
    for group_name, records in sorted(groups.items()):
        if len(records) < min_documents:
            output["groups"][group_name] = {
                "status": "skipped",
                "reason": f"requires at least {min_documents} documents",
                "document_count": len(records),
                "pooled_fallback": "rerun without --group-by-tool",
            }
            continue

        embedding_documents = [record["embedding_document"] for record in records]
        topic_documents = [record["topic_document"] for record in records]
        attack_documents = [record.get("attack_document") for record in records]
        unique_attack_documents = list(
            dict.fromkeys(
                document
                for document in attack_documents
                if isinstance(document, str) and document
            )
        )
        all_documents = [
            *embedding_documents,
            *topic_documents,
            *unique_attack_documents,
        ]
        all_embeddings = embedding_model.encode(
            all_documents,
            show_progress_bar=True,
        )
        document_count = len(records)
        embeddings = all_embeddings[:document_count]
        behavior_embeddings = all_embeddings[document_count : 2 * document_count]
        attack_embeddings = {
            document: all_embeddings[2 * document_count + index]
            for index, document in enumerate(unique_attack_documents)
        }
        attack_similarities = [
            (
                _cosine_similarity(behavior_embedding, attack_embeddings[attack_document])
                if isinstance(attack_document, str)
                and attack_document in attack_embeddings
                else None
            )
            for behavior_embedding, attack_document in zip(
                behavior_embeddings,
                attack_documents,
                strict=True,
            )
        ]
        model = BERTopic(
            embedding_model=embedding_model,
            vectorizer_model=CountVectorizer(stop_words=topic_stop_words),
            min_topic_size=max(2, min(min_topic_size, len(topic_documents))),
            calculate_probabilities=True,
            verbose=False,
        )
        topics, probabilities = model.fit_transform(topic_documents, embeddings=embeddings)
        topic_info = _compact_topic_info(
            model.get_topic_info().to_dict(orient="records")
        )
        model_id = (
            f"{group_name.replace(':', '__')}_{document_view}_attack_conditioned_v2"
        )
        assignments: list[dict[str, Any]] = []
        for index, (record, topic, attack_similarity) in enumerate(
            zip(records, topics, attack_similarities, strict=True)
        ):
            topic_id = int(topic)
            assignments.append(
                {
                    **{
                        key: value
                        for key, value in record.items()
                        if key
                        not in {"embedding_document", "topic_document", "attack_document"}
                    },
                    "model_id": model_id,
                    "topic_id": topic_id,
                    "topic_namespace": f"{group_name}/topic_{topic_id}",
                    "topic_membership_probability": _membership_probability(
                        probabilities, index
                    ),
                    "attack_similarity": attack_similarity,
                    "status": "provisional",
                    "human_review_status": "unreviewed",
                }
            )

        attack_candidate_topics = _rank_topics_by_attack_similarity(
            topic_info,
            assignments,
            group_name=group_name,
        )
        output["groups"][group_name] = {
            "status": "fitted",
            "model_id": model_id,
            "document_view": document_view,
            "embedding_input": "attack_context_plus_behavior",
            "topic_representation_input": "behavior_only",
            "embedding_model": embedding_model_name,
            "attack_similarity_metric": "cosine(payload_only, behavior_only)",
            "attack_topic_ranking_metric": "p90",
            "document_count": len(topic_documents),
            "topics": topic_info,
            "attack_candidate_topics": attack_candidate_topics,
            "assignments": assignments,
            "review_samples": build_review_samples(
                assignments,
                samples_per_bucket=samples_per_bucket,
                random_seed=random_seed,
            ),
        }
        if model_dir is not None:
            destination = model_dir / model_id
            destination.parent.mkdir(parents=True, exist_ok=True)
            model.save(destination, serialization="safetensors", save_ctfidf=True)

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("job_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--document-view", choices=("raw", "normalized"), default="normalized")
    parser.add_argument(
        "--group-by-tool",
        action="store_true",
        help="Split unit types by tool; pooled unit-type models are the default.",
    )
    parser.add_argument("--min-documents", type=int, default=10)
    parser.add_argument("--min-topic-size", type=int, default=10)
    parser.add_argument("--samples-per-bucket", type=int, default=3)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    groups = collect_documents(
        args.job_dir,
        document_view=args.document_view,
        group_by_tool=args.group_by_tool,
    )
    result = {
        "job_dir": str(args.job_dir),
        "output_schema_version": "v3",
        "document_storage": "referenced",
        "embedding_conditioning": "attack_context_plus_behavior",
        "topic_representation": "behavior_only",
        "attack_similarity_metric": "cosine(payload_only, behavior_only)",
        "discovery_stage": "provisional_topics",
        "document_view": args.document_view,
        "group_by_tool": args.group_by_tool,
        "input_document_count": sum(len(records) for records in groups.values()),
        **fit_topics(
            groups,
            document_view=args.document_view,
            min_documents=args.min_documents,
            min_topic_size=args.min_topic_size,
            model_dir=args.model_dir,
            samples_per_bucket=args.samples_per_bucket,
            random_seed=args.random_seed,
            embedding_model_name=args.embedding_model,
        ),
    }
    destination = args.output or args.job_dir / "attack_relation_topics.json"
    destination.write_text(json.dumps(result, indent=2, default=_json_default) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
