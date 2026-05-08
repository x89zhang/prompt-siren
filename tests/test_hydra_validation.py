# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Test Hydra app validation logic."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from prompt_siren.config.exceptions import ConfigValidationError
from prompt_siren.config.experiment_config import ExperimentConfig
from prompt_siren.config.export import export_default_config
from prompt_siren.hydra_app import (
    _ensure_local_swebench_images,
    _expand_entities_for_required_runs,
    validate_config,
)
from prompt_siren.registry_base import UnknownComponentError


class _Entity:
    def __init__(self, entity_id: str):
        self.id = entity_id


class _Persistence:
    def __init__(self):
        self.resume_only_partial = False
        self._run_counts = {}
        self._resume_run_ids = {}

    def get_run_counts(self):
        return self._run_counts

    def list_resume_run_ids(self, entity_id: str):
        return self._resume_run_ids.get(entity_id, [])


class _JobConfig:
    n_runs_per_task = 1


class _Job:
    def __init__(self):
        self.persistence = _Persistence()
        self.job_config = _JobConfig()


class TestExpandEntitiesForRequiredRuns:
    """Test job-run expansion for fresh and checkpoint resume scheduling."""

    def test_expands_to_required_fresh_run_count(self):
        job = _Job()
        job.job_config.n_runs_per_task = 3
        job.persistence._run_counts = {"task1": 1}
        task1 = _Entity("task1")
        task2 = _Entity("task2")

        expanded = _expand_entities_for_required_runs(job, [task1, task2])

        assert [entity.id for entity in expanded] == [
            "task1",
            "task1",
            "task2",
            "task2",
            "task2",
        ]

    def test_resume_only_partial_expands_only_incomplete_checkpoints(self):
        job = _Job()
        job.job_config.n_runs_per_task = 5
        job.persistence.resume_only_partial = True
        job.persistence._resume_run_ids = {"task1": ["checkpoint-a"], "task2": []}
        task1 = _Entity("task1")
        task2 = _Entity("task2")

        expanded = _expand_entities_for_required_runs(job, [task1, task2])

        assert [entity.id for entity in expanded] == ["task1"]

    def test_resume_only_partial_honors_repeated_checkpoint_ids(self):
        job = _Job()
        job.job_config.n_runs_per_task = 5
        job.persistence.resume_only_partial = True
        job.persistence._resume_run_ids = {
            "task1": ["checkpoint-a", "checkpoint-a", "checkpoint-a"]
        }
        task1 = _Entity("task1")

        expanded = _expand_entities_for_required_runs(job, [task1])

        assert [entity.id for entity in expanded] == ["task1", "task1", "task1"]


class TestEnsureLocalSwebenchImages:
    @pytest.mark.anyio
    async def test_use_cache_false_rebuilds_required_images_even_if_present(self):
        config = ExperimentConfig.model_validate(
            {
                "name": "test",
                "agent": {"type": "plain", "config": {"model": "test"}},
                "dataset": {
                    "type": "swebench",
                    "config": {"registry": None, "use_cache": False},
                },
                "attack": {"type": "template_string", "config": {}},
                "sandbox_manager": {"type": "local-docker", "config": {}},
            }
        )

        docker = SimpleNamespace(close=AsyncMock())
        builder = SimpleNamespace(build_all_specs=AsyncMock(return_value=[]))

        with (
            patch(
                "prompt_siren.hydra_app._get_required_swebench_image_tags",
                return_value={"siren-swebench-benign:sympy__sympy-15345"},
            ),
            patch("prompt_siren.hydra_app.create_docker_client_from_config", return_value=docker),
            patch(
                "prompt_siren.hydra_app.get_image_build_specs",
                return_value=[SimpleNamespace(tag="siren-swebench-benign:sympy__sympy-15345")],
            ),
            patch("prompt_siren.hydra_app.ImageBuilder", return_value=builder) as image_builder,
        ):
            await _ensure_local_swebench_images(config, [_Entity("task")])

        image_builder.assert_called_once()
        assert image_builder.call_args.kwargs["rebuild_existing"] is True
        builder.build_all_specs.assert_awaited_once()
        docker.close.assert_awaited_once()


@pytest.mark.usefixtures("mock_api_keys")
class TestValidateConfig:
    """Test the validate_config() function."""

    def test_valid_minimal_config(self):
        """Valid config should pass validation."""
        cfg = OmegaConf.create(
            {
                "name": "test",
                "agent": {"type": "plain", "config": {"model": "openai:gpt-4"}},
                "dataset": {
                    "type": "agentdojo",
                    "config": {"suite_name": "workspace", "version": "v1.2.2"},
                },
                "attack": None,
                "execution": {"concurrency": 1},
                "task_ids": None,
                "output": {"jobs_dir": "jobs"},
                "telemetry": {"trace_console": False},
                "usage_limits": None,
            }
        )

        result = validate_config(cfg, execution_mode="benign")
        assert result.name == "test"
        assert result.agent.type == "plain"

    def test_unknown_agent_type(self):
        """Unknown agent type should raise UnknownComponentError."""
        cfg = OmegaConf.create(
            {
                "name": "test",
                "agent": {"type": "fake_agent", "config": {}},
                "dataset": {
                    "type": "agentdojo",
                    "config": {"suite_name": "workspace", "version": "v1.2.2"},
                },
                "attack": None,
                "execution": {"concurrency": 1},
                "task_ids": None,
                "output": {"jobs_dir": "jobs"},
                "telemetry": {"trace_console": False},
                "usage_limits": None,
            }
        )

        with pytest.raises(UnknownComponentError):
            validate_config(cfg, execution_mode="benign")

    def test_invalid_agent_config_error_message(self):
        """Invalid agent config should include component type in error."""
        cfg = OmegaConf.create(
            {
                "name": "test",
                "agent": {
                    "type": "plain",
                    "config": {},
                },  # Missing required 'model' field
                "dataset": {
                    "type": "agentdojo",
                    "config": {"suite_name": "workspace", "version": "v1.2.2"},
                },
                "attack": None,
                "execution": {"concurrency": 1},
                "task_ids": None,
                "output": {"jobs_dir": "jobs"},
                "telemetry": {"trace_console": False},
                "usage_limits": None,
            }
        )

        with pytest.raises(ConfigValidationError, match=r"agent 'plain'") as exc_info:
            validate_config(cfg, execution_mode="benign")

        # Verify the structured exception attributes
        assert exc_info.value.component_type == "agent"
        assert exc_info.value.component_name == "plain"
        assert "model" in str(exc_info.value).lower()

    def test_invalid_dataset_config_error_message(self):
        """Invalid dataset config should include component type in error."""
        cfg = OmegaConf.create(
            {
                "name": "test",
                "agent": {"type": "plain", "config": {"model": "openai:gpt-4"}},
                "dataset": {
                    "type": "agentdojo",
                    "config": {"suite_name": 123},
                },  # Invalid type (int instead of str)
                "attack": None,
                "execution": {"concurrency": 1},
                "task_ids": None,
                "output": {"jobs_dir": "jobs"},
                "telemetry": {"trace_console": False},
                "usage_limits": None,
            }
        )

        with pytest.raises(ConfigValidationError, match=r"dataset 'agentdojo'") as exc_info:
            validate_config(cfg, execution_mode="benign")

        # Verify the structured exception attributes
        assert exc_info.value.component_type == "dataset"
        assert exc_info.value.component_name == "agentdojo"

    def test_invalid_attack_config_error_message(self):
        """Invalid attack config should include component type in error."""
        cfg = OmegaConf.create(
            {
                "name": "test",
                "agent": {"type": "plain", "config": {"model": "openai:gpt-4"}},
                "dataset": {
                    "type": "agentdojo",
                    "config": {"suite_name": "workspace", "version": "v1.2.2"},
                },
                "attack": {
                    "type": "template_string",
                    "config": {"attack_template": 123},
                },  # Invalid type (int instead of str)
                "execution": {"concurrency": 1},
                "task_ids": None,
                "output": {"jobs_dir": "jobs"},
                "telemetry": {"trace_console": False},
                "usage_limits": None,
            }
        )

        with pytest.raises(ConfigValidationError, match=r"attack 'template_string'") as exc_info:
            validate_config(cfg, execution_mode="attack")

        # Verify the structured exception attributes
        assert exc_info.value.component_type == "attack"
        assert exc_info.value.component_name == "template_string"

    def test_attack_mode_without_attack_raises(self):
        """execution_mode='attack' without attack should raise clear error."""
        cfg = OmegaConf.create(
            {
                "name": "test",
                "agent": {"type": "plain", "config": {"model": "openai:gpt-4"}},
                "dataset": {
                    "type": "agentdojo",
                    "config": {"suite_name": "workspace", "version": "v1.2.2"},
                },
                "attack": None,
                "execution": {"concurrency": 1},
                "task_ids": None,  # Attack mode but no attack!
                "output": {"jobs_dir": "jobs"},
                "telemetry": {"trace_console": False},
                "usage_limits": None,
            }
        )

        with pytest.raises(ValueError, match="Attack configuration is required for attack mode"):
            validate_config(cfg, execution_mode="attack")

    def test_benign_only_without_attack_is_valid(self):
        """benign_only mode without attack should be valid."""
        cfg = OmegaConf.create(
            {
                "name": "test",
                "agent": {"type": "plain", "config": {"model": "openai:gpt-4"}},
                "dataset": {
                    "type": "agentdojo",
                    "config": {"suite_name": "workspace", "version": "v1.2.2"},
                },
                "attack": None,
                "execution": {"concurrency": 1},
                "task_ids": None,
                "output": {"jobs_dir": "jobs"},
                "telemetry": {"trace_console": False},
                "usage_limits": None,
            }
        )

        result = validate_config(cfg, execution_mode="benign")
        assert result.attack is None


@pytest.mark.usefixtures("mock_api_keys")
class TestCliValidationIntegration:
    """Test CLI validation command integration with Hydra compose."""

    def test_validate_with_valid_config(self, tmp_path):
        """Test that CLI validation with valid config succeeds."""

        # Export default config to a temporary directory
        config_dir = tmp_path / "config"
        export_default_config(config_dir)

        # Initialize Hydra with the config directory (mimics _validate_config)
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            # Compose configuration with overrides
            cfg = compose(config_name="config", overrides=["+dataset=agentdojo-workspace"])

            # Validate directly (this is what _validate_config does)
            result = validate_config(cfg, execution_mode="benign")
            assert result is not None
            assert result.dataset.type == "agentdojo"

    def test_validate_with_invalid_config(self, tmp_path):
        """Test that CLI validation with invalid config raises error."""

        # Export default config to a temporary directory
        config_dir = tmp_path / "config"
        export_default_config(config_dir)

        # Initialize Hydra with the config directory
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            # Compose configuration with invalid agent type
            cfg = compose(
                config_name="config",
                overrides=[
                    "+dataset=agentdojo-workspace",
                    "agent.type=nonexistent_agent",
                ],
            )

            # Should raise UnknownComponentError
            with pytest.raises(UnknownComponentError):
                validate_config(cfg, execution_mode="benign")

    def test_validate_attack_mode_requires_attack(self, tmp_path):
        """Test that validating in attack mode without attack config fails."""

        # Export default config to a temporary directory
        config_dir = tmp_path / "config"
        export_default_config(config_dir)

        # Initialize Hydra with the config directory
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            # Compose configuration without attack
            cfg = compose(config_name="config", overrides=["+dataset=agentdojo-workspace"])

            # Should raise ValueError when validating in attack mode without attack
            with pytest.raises(
                ValueError,
                match="Attack configuration is required for attack mode",
            ):
                validate_config(cfg, execution_mode="attack")

    def test_validate_with_attack_config(self, tmp_path):
        """Test that CLI validation with attack config succeeds."""

        # Export default config to a temporary directory
        config_dir = tmp_path / "config"
        export_default_config(config_dir)

        # Initialize Hydra with the config directory
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            # Compose configuration with attack
            cfg = compose(
                config_name="config",
                overrides=[
                    "+dataset=agentdojo-workspace",
                    "+attack=agentdojo_important_instructions",
                ],
            )

            # Validate in attack mode
            result = validate_config(cfg, execution_mode="attack")
            assert result is not None
            assert result.attack is not None
            assert result.attack.type == "template_string"
