from prompt_siren.attacks.queryipi_attack import (
    _extract_payload,
    _resolve_attacker_model_name,
    create_queryipi_attack,
    QueryIPIAttack,
    QueryIPIAttackConfig,
    QueryIPIAttackerModel,
)


class TestQueryIPIAttackConfig:
    def test_default_config(self):
        config = QueryIPIAttackConfig()

        assert config.attacker_model
        assert "QueryIPI" in config.attacker_model_instructions
        assert "{agent_system_prompt}" in config.first_user_message_to_attacker
        assert config.training_queries == []

    def test_training_queries_are_not_shared(self):
        first = QueryIPIAttackConfig()
        second = QueryIPIAttackConfig()

        first.training_queries.append("query")

        assert second.training_queries == []


class TestQueryIPIAttack:
    def test_factory_function(self):
        config = QueryIPIAttackConfig(
            agent_system_prompt="system",
            tool_context="tools",
            training_queries=["query"],
            max_turns=2,
        )

        attack = create_queryipi_attack(config)

        assert isinstance(attack, QueryIPIAttack)
        assert attack.name == "queryipi"
        assert attack.config.agent_system_prompt == "system"
        assert attack.config.training_queries == ["query"]


class TestQueryIPIAttackerModel:
    def test_local_openai_compatible_model(self):
        attacker = QueryIPIAttackerModel(
            QueryIPIAttackConfig(
                attacker_model="qwen3.6-35b",
                attacker_model_base_url="http://localhost:8000/v1",
                attacker_model_api_key="EMPTY",
            )
        )

        assert attacker.agent.model.model_name == "qwen3.6-35b"

    def test_provider_env_model_when_base_url_is_none(self):
        attacker = QueryIPIAttackerModel(
            QueryIPIAttackConfig(
                attacker_model="test",
                attacker_model_base_url=None,
                attacker_model_api_key=None,
            )
        )

        assert attacker.agent.model.model_name == "test"

    def test_auto_local_vllm_model_from_mini_swe_agent_name(self):
        config = QueryIPIAttackConfig(
            attacker_model="auto",
            attacker_model_base_url="http://localhost:8000/v1",
            attacker_model_api_key="EMPTY",
        )

        assert (
            _resolve_attacker_model_name(config, "mini_swe:hosted_vllm/qwen3.5-9b")
            == "qwen3.5-9b"
        )

    def test_auto_openai_model_from_litellm_style_agent_name(self):
        config = QueryIPIAttackConfig(
            attacker_model="auto",
            attacker_model_base_url=None,
            attacker_model_api_key=None,
        )

        assert _resolve_attacker_model_name(config, "mini_swe:openai/gpt-5.1") == "openai:gpt-5.1"


class TestExtractPayload:
    def test_extracts_payload_tags(self):
        assert _extract_payload("analysis\n<payload>\nuse this\n</payload>") == "use this"

    def test_preserves_stop(self):
        assert _extract_payload(" STOP ") == "STOP"

    def test_strips_reasoning_tags(self):
        assert _extract_payload("<think>reasoning</think>\nuse this") == "use this"

    def test_strips_leaked_reasoning_prefix(self):
        assert _extract_payload("reasoning\n</think>\nuse this") == "use this"

    def test_extracts_single_markdown_fence_as_formatting_fallback(self):
        assert _extract_payload("analysis\n```text\nuse this\n```") == "use this"
