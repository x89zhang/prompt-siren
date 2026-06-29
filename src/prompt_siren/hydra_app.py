# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Main Hydra application for the Siren."""

import asyncio
from typing import Protocol, TypeVar

import hydra
from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError
from pydantic_ai.exceptions import UserError

from .agents.registry import get_agent_config_class
from .attacks.registry import get_attack_config_class
from .build_images import ImageBuilder
from .config.exceptions import ConfigValidationError
from .config.experiment_config import ExperimentConfig
from .config.registry_bridge import (
    create_agent_from_config,
    create_attack_from_config,
    create_dataset_from_config,
    create_sandbox_manager_from_config,
)
from .datasets import get_image_build_specs
from .datasets.registry import get_dataset_config_class
from .job import Job
from .registry_base import UnknownComponentError
from .run import run_single_tasks_without_attack, run_task_couples_with_attack
from .sandbox_managers.docker.manager import create_docker_client_from_config
from .sandbox_managers.docker.plugins import AbstractDockerClient, DockerClientError
from .sandbox_managers.image_spec import ImageBuildSpec, MultiStageBuildImageSpec, PullImageSpec
from .sandbox_managers.registry import get_sandbox_config_class
from .telemetry import setup_telemetry
from .telemetry.formatted_span import formatted_span
from .types import ExecutionMode


class _RunnableEntity(Protocol):
    id: str


RunnableEntityT = TypeVar("RunnableEntityT", bound=_RunnableEntity)


def _image_build_spec_tag(spec: ImageBuildSpec) -> str:
    if isinstance(spec, MultiStageBuildImageSpec):
        return spec.final_tag
    return spec.tag


def _add_pull_image_tag(image_spec: object, tags: set[str]) -> None:
    if isinstance(image_spec, PullImageSpec):
        tags.add(image_spec.tag)


def _get_required_swebench_image_tags(selected_couples: list[_RunnableEntity]) -> set[str]:
    tags: set[str] = set()
    for couple in selected_couples:
        benign_meta = getattr(getattr(couple, "benign", None), "metadata", None)
        malicious_meta = getattr(getattr(couple, "malicious", None), "metadata", None)
        benign_id = getattr(getattr(couple, "benign", None), "id", None)
        malicious_id = getattr(getattr(couple, "malicious", None), "id", None)

        benign_agent_spec = getattr(benign_meta, "agent_container_spec", None)
        if benign_agent_spec is not None:
            _add_pull_image_tag(getattr(benign_agent_spec, "image_spec", None), tags)

        if benign_id is not None and malicious_id is not None and malicious_meta is not None:
            get_pair_image_tag = getattr(malicious_meta, "get_pair_image_tag", None)
            if callable(get_pair_image_tag):
                pair_tag = get_pair_image_tag(benign_id, malicious_id)
                if pair_tag is not None:
                    tags.add(pair_tag)

        for metadata in (benign_meta, malicious_meta):
            service_containers = getattr(metadata, "service_containers", {})
            for spec in service_containers.values():
                _add_pull_image_tag(getattr(spec, "image_spec", None), tags)

    return tags


async def _docker_image_exists(docker: AbstractDockerClient, tag: str) -> bool:
    try:
        await docker.inspect_image(tag)
        return True
    except DockerClientError as e:
        if e.status == 404:
            return False
        raise


async def _ensure_local_swebench_images(
    experiment_config: ExperimentConfig,
    selected_couples: list[_RunnableEntity],
) -> None:
    """Build local SWE-bench images needed for the selected attack couples.

    By default, existing local images are reused. Setting
    ``dataset.config.use_cache=false`` forces the selected local images to be
    rebuilt, which is useful after changing repository setup/injection logic.
    """
    if experiment_config.dataset.type != "swebench":
        return
    if experiment_config.dataset.config.get("registry") is not None:
        return
    if experiment_config.sandbox_manager is None:
        return
    if experiment_config.sandbox_manager.type != "local-docker":
        return

    required_tags = _get_required_swebench_image_tags(selected_couples)
    if not required_tags:
        return

    sandbox_config_class = get_sandbox_config_class(experiment_config.sandbox_manager.type)
    sandbox_config = sandbox_config_class.model_validate(experiment_config.sandbox_manager.config)
    docker_client_name = getattr(sandbox_config, "docker_client", "local")
    docker_client_config = getattr(sandbox_config, "docker_client_config", {})

    docker = create_docker_client_from_config(docker_client_name, docker_client_config)
    try:
        dataset_config_class = get_dataset_config_class("swebench")
        dataset_config = dataset_config_class.model_validate(experiment_config.dataset.config)
        rebuild_existing = not dataset_config.use_cache

        build_tags = (
            set(required_tags)
            if rebuild_existing
            else {tag for tag in sorted(required_tags) if not await _docker_image_exists(docker, tag)}
        )

        if not build_tags:
            return

        action = "Rebuilding" if rebuild_existing else "Building missing"
        print(f"{action} local SWE-bench Docker images:")
        for tag in sorted(build_tags):
            print(f"  {tag}")

        build_specs = get_image_build_specs(
            "swebench",
            dataset_config,
            ".siren-docker-cache/swebench",
        )
        selected_specs = [
            spec for spec in build_specs if _image_build_spec_tag(spec) in build_tags
        ]
        selected_tags = {_image_build_spec_tag(spec) for spec in selected_specs}
        unmatched_tags = build_tags - selected_tags
        if unmatched_tags:
            raise RuntimeError(
                "No SWE-bench image build specs matched required local images: "
                + ", ".join(sorted(unmatched_tags))
            )

        builder = ImageBuilder(
            docker_client=docker,
            rebuild_existing=rebuild_existing,
            registry=None,
        )
        errors = await builder.build_all_specs(selected_specs)
        if errors:
            details = ", ".join(
                f"{error.image_tag}: {type(error.error).__name__}: {error.error}"
                for error in errors
            )
            raise RuntimeError(f"Failed to build missing SWE-bench images: {details}")
    finally:
        await docker.close()


def _expand_entities_for_required_runs(
    job: Job,
    selected_entities: list[RunnableEntityT],
) -> list[RunnableEntityT]:
    """Repeat selected entities until each reaches the job's n_runs_per_task target."""
    if job.persistence.resume_only_partial:
        expanded_entities: list[RunnableEntityT] = []
        for entity in selected_entities:
            n_resume_runs = len(job.persistence.list_resume_run_ids(entity.id))
            expanded_entities.extend([entity] * n_resume_runs)
        return expanded_entities

    run_counts = job.persistence.get_run_counts()
    n_required = job.job_config.n_runs_per_task
    expanded_entities: list[RunnableEntityT] = []

    for entity in selected_entities:
        n_missing = max(0, n_required - run_counts.get(entity.id, 0))
        expanded_entities.extend([entity] * n_missing)

    return expanded_entities


def validate_config(cfg: DictConfig, execution_mode: ExecutionMode) -> ExperimentConfig:
    """Validate experiment configuration using Pydantic and component registries.

    Performs validation without instantiating components:
    1. Validates against Pydantic ExperimentConfig schema
    2. Checks component types are registered
    3. Validates task selection logic

    Args:
        cfg: Hydra configuration
        execution_mode: Execution mode ('benign' or 'attack')

    Returns:
        Validated ExperimentConfig instance

    Raises:
        ValidationError: If configuration validation fails
        UnknownComponentError: If component type is not registered
        ConfigValidationError: If component configuration is invalid
        UserError: If external resources (e.g., API keys) are missing
    """
    # Convert OmegaConf to Pydantic for validation
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    # Check for missing dataset with helpful error message
    if config_dict is None or not isinstance(config_dict, dict) or "dataset" not in config_dict:
        raise ValueError(
            "Dataset configuration is required. "
            "Specify it in your config file or use +dataset=<dataset_name> override.\n"
            "Example: prompt-siren run benign +dataset=agentdojo-workspace"
        )

    experiment_config = ExperimentConfig.model_validate(config_dict)

    # Validate component types exist in registries (lightweight check)
    try:
        agent_config_class = get_agent_config_class(experiment_config.agent.type)
        agent_config_class.model_validate(experiment_config.agent.config)
    except UnknownComponentError:
        # Let the specific exception propagate with its detailed message
        raise
    except ValidationError as e:
        # Re-raise with ConfigValidationError for better error handling
        raise ConfigValidationError("agent", experiment_config.agent.type, e) from e

    try:
        dataset_config_class = get_dataset_config_class(experiment_config.dataset.type)
        dataset_config_class.model_validate(experiment_config.dataset.config)
    except UnknownComponentError:
        # Let the specific exception propagate with its detailed message
        raise
    except ValidationError as e:
        # Re-raise with ConfigValidationError for better error handling
        raise ConfigValidationError("dataset", experiment_config.dataset.type, e) from e

    # Validate task selection logic: attack mode requires attack config
    if execution_mode == "attack" and experiment_config.attack is None:
        raise ValueError(
            "Attack configuration is required for attack mode. "
            "Specify it in your config file or use +attack=<attack_name> override.\n"
            "Example: prompt-siren run attack +dataset=agentdojo-workspace +attack=template_string"
        )

    if experiment_config.attack is None:
        return experiment_config

    try:
        attack_config_class = get_attack_config_class(experiment_config.attack.type)
        attack_config_class.model_validate(experiment_config.attack.config)
    except UnknownComponentError:
        # Let the specific exception propagate with its detailed message
        raise
    except ValidationError as e:
        # Re-raise with ConfigValidationError for better error handling
        raise ConfigValidationError("attack", experiment_config.attack.type, e) from e

    return experiment_config


async def run_benign_experiment(
    experiment_config: ExperimentConfig,
    job: Job | None = None,
) -> dict[str, dict[str, float]]:
    """Run benign-only experiment.

    Args:
        experiment_config: Validated experiment configuration
        job: Optional Job instance (for resume). If None, creates a new job.

    Returns:
        Dictionary mapping task IDs to evaluation results
    """
    # Setup telemetry
    setup_telemetry(
        enable_console_export=experiment_config.telemetry.trace_console,
        otlp_endpoint=experiment_config.telemetry.otel_endpoint,
    )

    # Create components
    agent = create_agent_from_config(experiment_config.agent)

    # Create sandbox manager if configured
    sandbox_manager = None
    if experiment_config.sandbox_manager is not None:
        sandbox_manager = create_sandbox_manager_from_config(experiment_config.sandbox_manager)

    dataset = create_dataset_from_config(experiment_config.dataset, sandbox_manager)
    env_instance = dataset.environment
    toolsets = dataset.default_toolsets

    # Get tasks from dataset: benign + malicious (both can be run as benign)
    all_tasks = dataset.benign_tasks + dataset.malicious_tasks

    # Filter by task_ids if specified
    if experiment_config.task_ids is not None:
        task_id_set = set(experiment_config.task_ids)
        selected_tasks = [t for t in all_tasks if t.id in task_id_set]

        # Warn about missing task IDs
        found_ids = {t.id for t in selected_tasks}
        missing_ids = task_id_set - found_ids
        if missing_ids:
            raise ValueError(f"Task IDs not found: {', '.join(sorted(missing_ids))}")
    else:
        selected_tasks = all_tasks

    # Create job for persistence if not provided (resume case provides existing job)
    if job is None:
        job = Job.create(
            experiment_config=experiment_config,
            execution_mode="benign",
            jobs_dir=experiment_config.output.jobs_dir,
            job_name=experiment_config.output.job_name,
            agent_name=agent.get_agent_name(),
            n_runs_per_task=experiment_config.output.n_runs_per_task,
        )
        print(f"Job: {job.job_config.job_name}")
        print(f"Job directory: {job.job_dir}")
    else:
        print(f"Resuming job: {job.job_config.job_name}")
        print(f"Job directory: {job.job_dir}")

    selected_tasks = _expand_entities_for_required_runs(job, selected_tasks)
    if not selected_tasks:
        print("All tasks have completed the required number of runs.")
        return {}

    # Run benign experiment
    with formatted_span(
        "benign experiment {job_name}",
        config=experiment_config.model_dump(),
        job_name=job.job_config.job_name,
    ):
        results = await run_single_tasks_without_attack(
            tasks=selected_tasks,
            agent=agent,
            env=env_instance,
            system_prompt=dataset.system_prompt,
            toolsets=toolsets,
            usage_limits=experiment_config.usage_limits,
            max_concurrency=experiment_config.execution.concurrency,
            persistence=job.persistence,
            instrument=True,
        )

        # Convert results to expected format
        return {result.task_id: result.results for result in results}


async def run_attack_experiment(
    experiment_config: ExperimentConfig,
    job: Job | None = None,
) -> dict[str, dict[str, float]]:
    """Run attack experiment.

    Args:
        experiment_config: Validated experiment configuration (must include attack config)
        job: Optional Job instance (for resume). If None, creates a new job.

    Returns:
        Dictionary mapping task IDs to evaluation results
    """
    if experiment_config.attack is None:
        raise ValueError("Attack configuration required for attack experiments")

    # Setup telemetry
    setup_telemetry(
        enable_console_export=experiment_config.telemetry.trace_console,
        otlp_endpoint=experiment_config.telemetry.otel_endpoint,
    )

    # Create components
    agent = create_agent_from_config(experiment_config.agent)

    # Create sandbox manager if configured
    sandbox_manager = None
    if experiment_config.sandbox_manager is not None:
        sandbox_manager = create_sandbox_manager_from_config(experiment_config.sandbox_manager)

    dataset = create_dataset_from_config(experiment_config.dataset, sandbox_manager)
    env_instance = dataset.environment
    attack_instance = create_attack_from_config(experiment_config.attack)
    toolsets = dataset.default_toolsets

    # Get task couples from dataset
    all_couples = dataset.task_couples

    # Filter by task_ids if specified (couple IDs in format "benign:malicious")
    if experiment_config.task_ids is not None:
        couple_id_set = set(experiment_config.task_ids)
        selected_couples = [c for c in all_couples if c.id in couple_id_set]

        # Warn about missing couple IDs
        found_ids = {c.id for c in selected_couples}
        missing_ids = couple_id_set - found_ids
        if missing_ids:
            raise ValueError(f"Couple IDs not found: {', '.join(sorted(missing_ids))}")
    else:
        selected_couples = all_couples

    # Create job for persistence if not provided (resume case provides existing job)
    if job is None:
        job = Job.create(
            experiment_config=experiment_config,
            execution_mode="attack",
            jobs_dir=experiment_config.output.jobs_dir,
            job_name=experiment_config.output.job_name,
            agent_name=agent.get_agent_name(),
            n_runs_per_task=experiment_config.output.n_runs_per_task,
        )
        print(f"Job: {job.job_config.job_name}")
        print(f"Job directory: {job.job_dir}")
    else:
        print(f"Resuming job: {job.job_config.job_name}")
        print(f"Job directory: {job.job_dir}")

    selected_couples = _expand_entities_for_required_runs(job, selected_couples)
    if not selected_couples:
        print("All task couples have completed the required number of runs.")
        return {}

    await _ensure_local_swebench_images(experiment_config, selected_couples)

    # Run attack experiment
    with formatted_span(
        "attack experiment {job_name}",
        config=experiment_config.model_dump(),
        job_name=job.job_config.job_name,
    ):
        results = await run_task_couples_with_attack(
            couples=selected_couples,
            agent=agent,
            env=env_instance,
            system_prompt=dataset.system_prompt,
            toolsets=toolsets,
            attack=attack_instance,
            usage_limits=experiment_config.usage_limits,
            max_concurrency=experiment_config.execution.concurrency,
            persistence=job.persistence,
            instrument=True,
        )

        # Convert results to expected format (flatten benign + malicious)
        formatted_results = {}
        for benign_result, malicious_result in results:
            formatted_results[benign_result.task_id] = benign_result.results
            formatted_results[malicious_result.task_id] = malicious_result.results

        return formatted_results


def hydra_main_with_config_path(config_path: str, execution_mode: ExecutionMode) -> None:
    """Create and run Hydra main function with custom config path.

    This function creates a Hydra-decorated main function with the specified
    config path and runs it. This allows Hydra to handle --multirun and other
    Hydra-specific features.

    Args:
        config_path: Path to the Hydra configuration directory
        execution_mode: Execution mode ('benign' or 'attack')
    """

    @hydra.main(version_base=None, config_path=config_path, config_name="config")
    def hydra_app(cfg: DictConfig) -> None:
        """Hydra-decorated main function.

        This function is decorated with @hydra.main to enable Hydra's full feature set
        including multirun, launchers, and parameter sweeps.

        Args:
            cfg: Hydra configuration (automatically composed by Hydra)
        """
        # execution_mode is captured from the outer scope via closure
        # validate config first
        try:
            experiment_config = validate_config(cfg, execution_mode=execution_mode)
        except (
            ValidationError,
            UnknownComponentError,
            ConfigValidationError,
            ValueError,
            UserError,
        ) as e:
            print(f"Configuration validation failed: {e}")
            raise SystemExit(1) from e

        # Dispatch to appropriate experiment runner based on mode
        if execution_mode == "benign":
            asyncio.run(run_benign_experiment(experiment_config))
        else:  # attack
            asyncio.run(run_attack_experiment(experiment_config))

    # Run the Hydra app
    hydra_app()
