# Copyright (c) Meta Platforms, Inc. and affiliates.
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from prompt_siren.sandbox_managers.abstract import ExecOutput
from prompt_siren.sandbox_managers.apptainer import (
    ApptainerSandboxConfig,
    ApptainerSandboxManager,
    create_apptainer_sandbox_manager,
    image_tag_to_sif_name,
)
from prompt_siren.sandbox_managers.apptainer.manager import (
    ApptainerBatchState,
    ApptainerContainer,
)
from prompt_siren.sandbox_managers.image_spec import PullImageSpec
from prompt_siren.sandbox_managers.sandbox_task_setup import (
    ContainerSetup,
    ContainerSpec,
    TaskSetup,
)

pytestmark = pytest.mark.anyio


def test_image_tag_to_sif_name_is_stable() -> None:
    assert (
        image_tag_to_sif_name(
            "ghcr.io/ethz-spylab/prompt-siren-images/siren-swebench-benign:astropy__astropy-14995"
        )
        == "ghcr.io__ethz-spylab__prompt-siren-images__siren-swebench-benign__astropy__astropy-14995.sif"
    )


def test_factory_creates_manager() -> None:
    config = ApptainerSandboxConfig(image_dir=Path("/tmp/images"))
    manager = create_apptainer_sandbox_manager(config)
    assert isinstance(manager, ApptainerSandboxManager)
    assert manager.config == config


async def test_setup_task_starts_agent_and_service_instances(tmp_path: Path) -> None:
    agent_image = tmp_path / "agent.sif"
    service_image = tmp_path / "service.sif"
    agent_image.touch()
    service_image.touch()
    manager = ApptainerSandboxManager(
        ApptainerSandboxConfig(
            work_dir=tmp_path,
            image_overrides={
                "agent:latest": agent_image,
                "service:latest": service_image,
            },
        )
    )
    agent = ContainerSetup(
        name="agent",
        spec=ContainerSpec(image_spec=PullImageSpec(tag="agent:latest")),
    )
    service = ContainerSetup(
        name="service",
        spec=ContainerSpec(
            image_spec=PullImageSpec(tag="service:latest"),
            hostname="svc.local",
            command=["python3", "/server.py"],
        ),
    )
    task_setup = TaskSetup(
        task_id="test",
        agent_container=agent,
        service_containers={"service": service},
    )

    with patch("shutil.which", return_value="/usr/bin/apptainer"):
        with patch.object(manager, "_create_overlay", new=AsyncMock()):
            with patch.object(
                manager,
                "_run",
                new=AsyncMock(return_value=ExecOutput(outputs=[], exit_code=0)),
            ) as run_mock:
                async with manager.setup_batch([task_setup]):
                    async with manager.setup_task(task_setup) as state:
                        assert state.agent_container_id.endswith(":agent")
                        assert state.service_containers["service"].endswith(":service")
                        assert state.network_id is None

    run_commands = [call.args[0] for call in run_mock.call_args_list]
    assert any(cmd[:3] == ["apptainer", "instance", "start"] for cmd in run_commands)
    assert any(
        "nohup python3 /server.py" in " ".join(cmd)
        for cmd in run_commands
        if len(cmd) > 1 and cmd[1] == "exec"
    )
    assert any(
        cmd[-3] == "/bin/sh"
        for cmd in run_commands
        if len(cmd) > 1 and cmd[1] == "exec"
    )
    assert any(cmd[:3] == ["apptainer", "instance", "stop"] for cmd in run_commands)


async def test_clone_sandbox_state_copies_agent_and_service_overlays(tmp_path: Path) -> None:
    agent_image = tmp_path / "agent.sif"
    service_image = tmp_path / "service.sif"
    agent_image.touch()
    service_image.touch()
    manager = ApptainerSandboxManager(
        ApptainerSandboxConfig(
            work_dir=tmp_path,
            image_overrides={
                "agent:latest": agent_image,
                "service:latest": service_image,
            },
        )
    )
    agent = ContainerSetup(
        name="agent",
        spec=ContainerSpec(image_spec=PullImageSpec(tag="agent:latest")),
    )
    service = ContainerSetup(
        name="service",
        spec=ContainerSpec(
            image_spec=PullImageSpec(tag="service:latest"),
            command=["python3", "/server.py"],
        ),
    )
    task_setup = TaskSetup(
        task_id="test",
        agent_container=agent,
        service_containers={"service": service},
    )

    async def create_overlay(path: Path) -> None:
        path.touch()

    with patch("shutil.which", return_value="/usr/bin/apptainer"):
        with patch.object(manager, "_create_overlay", new=AsyncMock(side_effect=create_overlay)):
            with patch.object(
                manager,
                "_run",
                new=AsyncMock(return_value=ExecOutput(outputs=[], exit_code=0)),
            ):
                async with manager.setup_batch([task_setup]):
                    async with manager.setup_task(task_setup) as state:
                        clone = await manager.clone_sandbox_state(state)
                        assert clone.execution_id != state.execution_id
                        assert clone.agent_container_id.endswith(":agent")
                        assert clone.service_containers["service"].endswith(":service")
                        await manager._cleanup_container(clone.agent_container_id)
                        await manager._cleanup_container(clone.service_containers["service"])


async def test_setup_batch_requires_sif_image(tmp_path: Path) -> None:
    manager = ApptainerSandboxManager(ApptainerSandboxConfig(image_dir=tmp_path))
    agent = ContainerSetup(
        name="agent",
        spec=ContainerSpec(image_spec=PullImageSpec(tag="missing:latest")),
    )
    task_setup = TaskSetup(task_id="test", agent_container=agent, service_containers={})

    with patch("shutil.which", return_value="/usr/bin/apptainer"):
        with pytest.raises(FileNotFoundError, match="No SIF image found"):
            async with manager.setup_batch([task_setup]):
                pass


async def test_exec_outside_batch_raises_error() -> None:
    manager = ApptainerSandboxManager(ApptainerSandboxConfig())
    with pytest.raises(RuntimeError, match="exec called outside of setup_batch context"):
        await manager.exec("container", "echo test")


@pytest.mark.parametrize(
    ("minimum_exec_timeout", "requested_timeout", "expected_timeout"),
    [(0, 5, 5), (30, 5, 30), (30, 60, 60), (30, None, 300)],
)
async def test_exec_applies_configured_minimum_timeout(
    tmp_path: Path,
    minimum_exec_timeout: int,
    requested_timeout: int | None,
    expected_timeout: int,
) -> None:
    manager = ApptainerSandboxManager(
        ApptainerSandboxConfig(minimum_exec_timeout=minimum_exec_timeout)
    )
    container = ApptainerContainer(
        container_id="container",
        instance_name="instance",
        image_path=tmp_path / "image.sif",
        overlay_path=tmp_path / "overlay.img",
    )
    manager._batch_state = ApptainerBatchState(
        batch_id="batch",
        containers={container.container_id: container},
    )

    with patch.object(
        manager,
        "_run",
        new=AsyncMock(return_value=ExecOutput(outputs=[], exit_code=0)),
    ) as run_mock:
        await manager.exec(container.container_id, ["true"], timeout=requested_timeout)

    assert run_mock.await_args.kwargs["timeout"] == expected_timeout
