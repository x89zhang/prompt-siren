from prompt_siren.attacks.queryipi_attack import (
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
