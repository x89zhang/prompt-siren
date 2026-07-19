import json
import sys
from pathlib import Path
from types import ModuleType
from typing import ClassVar

from scripts.cluster_attack_relations import (
    _behavior_document,
    _compact_topic_info,
    build_review_samples,
    collect_documents,
    fit_topics,
)


def _write_units(job_dir: Path) -> None:
    run_dir = job_dir / "task" / "run"
    run_dir.mkdir(parents=True)
    payload = {
        "task_id": "task",
        "run_id": "run",
        "attack_context": [{"content": "payload objective"}],
        "units": [
            {
                "unit_id": "agent-1",
                "unit_type": "agent_message",
                "trajectory_id": "task/run",
                "raw_document": "[ATTACK_CONTEXT]\nraw attack\n[UNIT=AGENT_MESSAGE]\nraw agent",
                "normalized_document": (
                    "[ATTACK_CONTEXT]\nnormalized attack\n"
                    "[UNIT=AGENT_MESSAGE]\nnormalized agent"
                ),
                "tool_name": "bash",
            },
            {
                "unit_id": "observation-1",
                "unit_type": "observation",
                "trajectory_id": "task/run",
                "raw_document": "raw observation",
                "normalized_document": "normalized observation",
                "tool_name": "bash",
            },
            {
                "unit_id": "transition-1",
                "unit_type": "observation_to_agent_transition",
                "trajectory_id": "task/run",
                "raw_document": "raw transition",
                "normalized_document": "normalized transition",
                "tool_name": "bash",
                "message_indices": [2, 3],
            },
        ],
    }
    (run_dir / "attack_analysis_units.json").write_text(json.dumps(payload))


def test_collect_documents_keeps_all_units_and_groups_by_unit_type(
    tmp_path: Path,
) -> None:
    _write_units(tmp_path)

    groups = collect_documents(tmp_path, document_view="normalized")

    assert set(groups) == {
        "agent_message",
        "observation",
        "observation_to_agent_transition",
    }
    assert sum(map(len, groups.values())) == 3
    agent = groups["agent_message"][0]
    assert "normalized attack" in agent["embedding_document"]
    assert agent["topic_document"] == "[UNIT=AGENT_MESSAGE]\nnormalized agent"
    assert agent["attack_document"] == "payload objective"
    transition = groups["observation_to_agent_transition"][0]
    assert transition["from_message_id"] == "m2"
    assert transition["to_message_id"] == "m3"
    assert "raw_document" not in transition
    assert "normalized_document" not in transition


def test_behavior_document_falls_back_when_unit_marker_is_missing() -> None:
    assert _behavior_document("plain behavior") == "plain behavior"


def test_tool_grouping_is_explicit_opt_in(tmp_path: Path) -> None:
    _write_units(tmp_path)

    groups = collect_documents(
        tmp_path,
        document_view="raw",
        group_by_tool=True,
    )

    assert set(groups) == {
        "agent_message:bash",
        "observation:bash",
        "observation_to_agent_transition:bash",
    }


def test_review_samples_keep_confidence_separate_and_include_outliers() -> None:
    assignments = [
        {
            "unit_id": "high",
            "unit_type": "agent_message",
            "topic_id": 1,
            "topic_namespace": "agent_message/topic_1",
            "topic_membership_probability": 0.9,
            "raw_document": "must not be copied",
            "normalized_document": "must not be copied",
        },
        {
            "unit_id": "boundary",
            "unit_type": "agent_message",
            "topic_id": 1,
            "topic_namespace": "agent_message/topic_1",
            "topic_membership_probability": 0.2,
        },
        {
            "unit_id": "outlier",
            "unit_type": "agent_message",
            "topic_id": -1,
            "topic_namespace": "agent_message/topic_-1",
            "topic_membership_probability": 0.0,
        },
    ]

    samples = build_review_samples(
        assignments,
        samples_per_bucket=1,
        random_seed=42,
    )

    assert samples["topics"]["1"]["representative"][0]["unit_id"] == "high"
    assert samples["topics"]["1"]["boundary"][0]["unit_id"] == "boundary"
    assert samples["outliers"][0]["unit_id"] == "outlier"
    assert "raw_document" not in samples["topics"]["1"]["representative"][0]
    assert "normalized_document" not in samples["topics"]["1"]["representative"][0]
    assert "judge_self_reported_confidence" not in assignments[0]


def test_topic_info_does_not_copy_representative_documents() -> None:
    compact = _compact_topic_info(
        [
            {
                "Topic": 1,
                "Count": 3,
                "Name": "topic name",
                "Representative_Docs": ["large document"],
            }
        ]
    )

    assert compact == [{"Topic": 1, "Count": 3, "Name": "topic name"}]


def test_fit_topics_uses_conditioned_embeddings_and_behavior_for_topic_words(
    monkeypatch,
) -> None:
    class FakeTopicInfo:
        def to_dict(self, *, orient: str):
            assert orient == "records"
            return [
                {
                    "Topic": 0,
                    "Count": 1,
                    "Name": "behavior_topic",
                    "Representative_Docs": ["copied document"],
                }
            ]

    class FakeSentenceTransformer:
        encoded_documents: ClassVar[list[str]] = []

        def __init__(self, model_name: str):
            assert model_name == "test-embedding-model"

        def encode(self, documents: list[str], *, show_progress_bar: bool):
            assert show_progress_bar is True
            self.encoded_documents = documents
            FakeSentenceTransformer.encoded_documents = documents
            return [[1.0, 1.0], [1.0, 0.0], [1.0, 0.0]]

    class FakeBERTopic:
        fitted_documents: ClassVar[list[str]] = []

        def __init__(self, **kwargs):
            assert kwargs["embedding_model"] is not None
            assert kwargs["vectorizer_model"].stop_words is not None

        def fit_transform(self, documents: list[str], *, embeddings):
            FakeBERTopic.fitted_documents = documents
            assert embeddings == [[1.0, 1.0]]
            return [0], [0.9]

        def get_topic_info(self):
            return FakeTopicInfo()

    bertopic_module = ModuleType("bertopic")
    bertopic_module.BERTopic = FakeBERTopic
    sentence_transformers_module = ModuleType("sentence_transformers")
    sentence_transformers_module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "bertopic", bertopic_module)
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers_module)

    output = fit_topics(
        {
            "agent_message": [
                {
                    "embedding_document": "attack objective plus behavior",
                    "topic_document": "behavior only",
                    "attack_document": "payload objective",
                    "unit_id": "agent-message:m1",
                }
            ]
        },
        document_view="normalized",
        min_documents=1,
        min_topic_size=2,
        model_dir=None,
        samples_per_bucket=1,
        random_seed=42,
        embedding_model_name="test-embedding-model",
    )

    group = output["groups"]["agent_message"]
    assert FakeSentenceTransformer.encoded_documents == [
        "attack objective plus behavior",
        "behavior only",
        "payload objective",
    ]
    assert FakeBERTopic.fitted_documents == ["behavior only"]
    assert group["embedding_input"] == "attack_context_plus_behavior"
    assert group["topic_representation_input"] == "behavior_only"
    assert "Representative_Docs" not in group["topics"][0]
    assert "embedding_document" not in group["assignments"][0]
    assert "topic_document" not in group["assignments"][0]
    assert "attack_document" not in group["assignments"][0]
    assert group["assignments"][0]["attack_similarity"] == 1.0
    assert group["attack_candidate_topics"][0]["topic_id"] == 0
    assert group["attack_candidate_topics"][0]["rank"] == 1
