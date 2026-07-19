import json
from pathlib import Path

from scripts.render_attack_relation_review import (
    render_attack_candidate_report,
    render_review_report,
)


def test_render_review_report_resolves_compact_references(tmp_path: Path) -> None:
    run_dir = tmp_path / "task" / "run"
    run_dir.mkdir(parents=True)
    source_file = run_dir / "attack_analysis_units.json"
    source_file.write_text(
        json.dumps(
            {
                "units": [
                    {
                        "unit_id": "transition:m2->m3",
                        "raw_document": "raw observation and reaction",
                        "normalized_document": "normalized observation and reaction",
                    }
                ]
            }
        )
    )
    reference = {
        "trajectory_id": "task/run",
        "unit_id": "transition:m2->m3",
        "message_indices": [2, 3],
        "from_message_id": "m2",
        "to_message_id": "m3",
        "source_file": "task/run/attack_analysis_units.json",
        "topic_id": 0,
        "topic_membership_probability": 0.8,
    }
    topic_file = tmp_path / "attack_relation_topics_v2.json"
    topic_file.write_text(
        json.dumps(
            {
                "job_dir": str(tmp_path),
                "document_view": "normalized",
                "input_document_count": 1,
                "groups": {
                    "observation_to_agent_transition": {
                        "status": "fitted",
                        "document_count": 1,
                        "model_id": "transition_normalized_v1",
                        "topics": [
                            {
                                "Topic": 0,
                                "Count": 1,
                                "Name": "0_reaction",
                                "Representation": ["reaction"],
                            }
                        ],
                        "review_samples": {
                            "topics": {
                                "0": {
                                    "representative": [reference],
                                    "boundary": [],
                                    "random": [],
                                }
                            },
                            "outliers": [],
                        },
                    }
                },
            }
        )
    )

    report = render_review_report(topic_file)

    assert "Group: `observation_to_agent_transition`" in report
    assert "Topic 0: `0_reaction` (1 units)" in report
    assert "| 0 | 1 | `0_reaction` | reaction |" in report
    assert "<details>" in report
    assert "Messages: `m2 → m3`" in report
    assert "normalized observation and reaction" in report
    assert "Proposed label:" in report
    assert "keep | merge | split | reject" in report


def test_render_review_report_truncates_long_documents(tmp_path: Path) -> None:
    run_dir = tmp_path / "task" / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "attack_analysis_units.json").write_text(
        json.dumps(
            {
                "units": [
                    {
                        "unit_id": "agent-message:m1",
                        "normalized_document": "start" + "x" * 200 + "end",
                    }
                ]
            }
        )
    )
    topic_file = tmp_path / "topics.json"
    topic_file.write_text(
        json.dumps(
            {
                "job_dir": str(tmp_path),
                "document_view": "normalized",
                "groups": {
                    "agent_message": {
                        "status": "fitted",
                        "topics": [{"Topic": 0, "Count": 1, "Name": "topic"}],
                        "review_samples": {
                            "topics": {
                                "0": {
                                    "representative": [
                                        {
                                            "unit_id": "agent-message:m1",
                                            "source_file": "task/run/attack_analysis_units.json",
                                        }
                                    ]
                                }
                            }
                        },
                    }
                },
            }
        )
    )

    report = render_review_report(topic_file, max_document_chars=80)

    assert "middle truncated for review report" in report
    assert "start" in report
    assert "end" in report


def test_candidate_report_labels_messages_from_top_ranked_topics(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "task" / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "attack_analysis_units.json").write_text(
        json.dumps(
            {
                "units": [
                    {
                        "unit_id": "agent-message:m3",
                        "normalized_document": "candidate behavior",
                    },
                    {
                        "unit_id": "agent-message:m5",
                        "normalized_document": "lower-ranked behavior",
                    },
                ]
            }
        )
    )
    topic_file = tmp_path / "topics.json"
    topic_file.write_text(
        json.dumps(
            {
                "job_dir": str(tmp_path),
                "document_view": "normalized",
                "groups": {
                    "agent_message": {
                        "status": "fitted",
                        "attack_candidate_topics": [
                            {
                                "rank": 1,
                                "topic_id": 4,
                                "count": 1,
                                "name": "payload_near",
                                "attack_similarity": {
                                    "p90": 0.9,
                                    "mean": 0.8,
                                    "median": 0.8,
                                    "max": 0.95,
                                },
                            },
                            {
                                "rank": 2,
                                "topic_id": 7,
                                "count": 1,
                                "name": "payload_farther",
                                "attack_similarity": {
                                    "p90": 0.5,
                                    "mean": 0.4,
                                    "median": 0.4,
                                    "max": 0.6,
                                },
                            },
                        ],
                        "assignments": [
                            {
                                "trajectory_id": "task/run",
                                "unit_id": "agent-message:m3",
                                "message_indices": [3],
                                "source_file": "task/run/attack_analysis_units.json",
                                "topic_id": 4,
                                "attack_similarity": 0.92,
                            },
                            {
                                "trajectory_id": "task/run",
                                "unit_id": "agent-message:m5",
                                "message_indices": [5],
                                "source_file": "task/run/attack_analysis_units.json",
                                "topic_id": 7,
                                "attack_similarity": 0.5,
                            },
                        ],
                    }
                },
            }
        )
    )

    report = render_attack_candidate_report(topic_file, top_topics_per_group=1)

    assert "Rank 1: Topic 4" in report
    assert "Trajectory `task/run`" in report
    assert "m3 · agent-message:m3" in report
    assert "candidate behavior" in report
    assert "Human message label:" in report
    assert "Rank 2: Topic 7" not in report
    assert "lower-ranked behavior" not in report
