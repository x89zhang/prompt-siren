# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Apptainer-based sandbox manager implementation.

This backend is separate from the Docker backend. It runs each configured
container as an Apptainer instance backed by a writable overlay, so files changed
by repeated exec calls persist for the lifetime of a task.
"""

from __future__ import annotations

import asyncio
import os
import re
import shlex
import shutil
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field

from ..abstract import ExecOutput, ExecTimeoutError, StderrChunk, StdoutChunk
from ..image_spec import PullImageSpec
from ..sandbox_state import ContainerID, SandboxState
from ..sandbox_task_setup import ContainerSetup, SandboxTaskSetup


def image_tag_to_sif_name(tag: str) -> str:
    """Map a Docker image tag to a stable SIF filename."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "__", tag).strip("_")
    return f"{safe}.sif"


class ApptainerSandboxConfig(BaseModel):
    """Configuration for the Apptainer sandbox manager."""

    apptainer_executable: str = Field(default="apptainer")
    image_dir: Path = Field(default=Path("sif-images"))
    work_dir: Path = Field(default=Path(".siren-apptainer"))
    overlay_size_mb: int = Field(default=4096, gt=0)
    bind_paths: tuple[Path, ...] = Field(default_factory=tuple)
    contain: bool = True
    cleanenv: bool = True
    fakeroot: bool = False
    nv: bool = False
    network_enabled: bool = Field(
        default=False,
        description=(
            "Reserved for config compatibility. Apptainer does not create Docker bridge "
            "networks; service instances use the host network namespace unless your "
            "cluster enables Apptainer network options externally."
        ),
    )
    batch_id_prefix: str = "workbench-apptainer"
    image_overrides: dict[str, Path] = Field(default_factory=dict)
    service_start_timeout: int = Field(default=30, gt=0)
    minimum_exec_timeout: int = Field(
        default=0,
        ge=0,
        description=(
            "Minimum timeout in seconds for Apptainer exec calls. A value of 0 "
            "preserves caller-provided timeouts without a floor."
        ),
    )


@dataclass
class ApptainerContainer:
    container_id: ContainerID
    instance_name: str
    image_path: Path
    overlay_path: Path
    environment: dict[str, str] = field(default_factory=dict)
    hostname: str | None = None
    startup_command: str | list[str] | None = None
    hosts_path: Path | None = None


@dataclass
class ApptainerBatchState:
    batch_id: str
    containers: dict[ContainerID, ApptainerContainer] = field(default_factory=dict)
    execution_containers: dict[str, set[ContainerID]] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class ApptainerSandboxManager:
    """Apptainer implementation of AbstractSandboxManager."""

    def __init__(self, config: ApptainerSandboxConfig):
        self._config = config
        self._batch_state: ApptainerBatchState | None = None

    @property
    def config(self) -> ApptainerSandboxConfig:
        return self._config

    @asynccontextmanager
    async def setup_batch(self, task_setups: Sequence[SandboxTaskSetup]) -> AsyncIterator[None]:
        if shutil.which(self._config.apptainer_executable) is None:
            raise RuntimeError(
                f"Apptainer executable '{self._config.apptainer_executable}' was not found"
            )

        for task_setup in task_setups:
            self._validate_task_setup(task_setup)
            self._resolve_image_path(task_setup.agent_container)
            for service_setup in task_setup.service_containers.values():
                self._resolve_image_path(service_setup)

        batch_id = f"{self._config.batch_id_prefix}-{uuid.uuid4().hex[:8]}"
        self._config.work_dir.mkdir(parents=True, exist_ok=True)
        self._batch_state = ApptainerBatchState(batch_id=batch_id)
        try:
            yield
        finally:
            await self._cleanup_batch()
            self._batch_state = None

    @asynccontextmanager
    async def setup_task(self, task_setup: SandboxTaskSetup) -> AsyncIterator[SandboxState]:
        if self._batch_state is None:
            raise RuntimeError("setup_task called outside of setup_batch context")

        self._validate_task_setup(task_setup)
        execution_id = uuid.uuid4().hex
        started_ids: list[ContainerID] = []

        try:
            service_hostnames = tuple(
                service_setup.spec.hostname
                for service_setup in task_setup.service_containers.values()
                if service_setup.spec.hostname
            )
            agent_container = await self._start_container(
                task_setup.agent_container,
                execution_id=execution_id,
                role_name="agent",
                host_aliases=service_hostnames,
            )
            started_ids.append(agent_container.container_id)

            service_ids: dict[str, ContainerID] = {}
            for service_name, service_setup in task_setup.service_containers.items():
                service_container = await self._start_container(
                    service_setup,
                    execution_id=execution_id,
                    role_name=service_name,
                )
                started_ids.append(service_container.container_id)
                service_ids[service_name] = service_container.container_id
                if service_setup.spec.command:
                    await self._start_background_command(service_container)

            yield SandboxState(
                agent_container_id=agent_container.container_id,
                service_containers=service_ids,
                execution_id=execution_id,
                network_id=task_setup.network_config.name if task_setup.network_config else None,
            )
        finally:
            await asyncio.gather(
                *(self._cleanup_container(container_id) for container_id in reversed(started_ids)),
                return_exceptions=True,
            )

    async def clone_sandbox_state(self, source_state: SandboxState) -> SandboxState:
        if self._batch_state is None:
            raise RuntimeError("clone_sandbox_state called outside of setup_batch context")

        new_execution_id = uuid.uuid4().hex
        new_agent = await self._clone_container(
            source_state.agent_container_id,
            execution_id=new_execution_id,
            role_name="agent",
        )
        new_services: dict[str, ContainerID] = {}
        try:
            for service_name, service_id in source_state.service_containers.items():
                cloned_service = await self._clone_container(
                    service_id,
                    execution_id=new_execution_id,
                    role_name=service_name,
                )
                new_services[service_name] = cloned_service.container_id
                if cloned_service.startup_command:
                    await self._start_background_command(cloned_service)
        except Exception:
            await self._cleanup_container(new_agent.container_id)
            await asyncio.gather(
                *(self._cleanup_container(container_id) for container_id in new_services.values()),
                return_exceptions=True,
            )
            raise

        return SandboxState(
            agent_container_id=new_agent.container_id,
            service_containers=new_services,
            execution_id=new_execution_id,
            network_id=source_state.network_id,
        )

    async def exec(
        self,
        container_id: ContainerID,
        cmd: str | list[str],
        stdin: str | bytes | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | None = None,
        timeout: int | None = None,
        shell_path: Path | None = None,
    ) -> ExecOutput:
        if self._batch_state is None:
            raise RuntimeError("exec called outside of setup_batch context")
        if user:
            raise NotImplementedError("ApptainerSandboxManager does not yet support per-exec user")

        async with self._batch_state._lock:
            container = self._batch_state.containers.get(container_id)
        if container is None:
            raise ValueError(f"Unknown Apptainer container id: {container_id}")

        timeout_value = max(
            timeout if timeout is not None else 300,
            self._config.minimum_exec_timeout,
        )
        shell = str(shell_path) if shell_path is not None else "/bin/bash"
        bash_cmd = self._normalize_shell_command(cmd)
        if cwd:
            bash_cmd = f"cd {shlex.quote(cwd)} && {bash_cmd}"

        full_cmd = [
            self._config.apptainer_executable,
            "exec",
            *self._exec_flags(env),
            f"instance://{container.instance_name}",
            shell,
            "-c",
            bash_cmd,
        ]
        return await self._run(
            full_cmd,
            stdin=stdin,
            timeout=timeout_value,
            container_id=container_id,
            original_cmd=cmd,
        )

    async def _start_container(
        self,
        container_setup: ContainerSetup,
        execution_id: str,
        role_name: str,
        overlay_source: Path | None = None,
        host_aliases: Sequence[str] = (),
    ) -> ApptainerContainer:
        if self._batch_state is None:
            raise RuntimeError("setup_task called outside of setup_batch context")

        image_path = self._resolve_image_path(container_setup)
        container_id = f"{execution_id}:{role_name}"
        instance_name = self._instance_name(execution_id, role_name)
        overlay_path = self._config.work_dir / f"{self._batch_state.batch_id}-{execution_id}-{role_name}.img"

        if overlay_source is None:
            await self._create_overlay(overlay_path)
        else:
            shutil.copy2(overlay_source, overlay_path)

        hosts_path = self._create_hosts_file(execution_id, role_name, host_aliases)
        container = ApptainerContainer(
            container_id=container_id,
            instance_name=instance_name,
            image_path=image_path,
            overlay_path=overlay_path,
            environment=container_setup.spec.environment or {},
            hostname=container_setup.spec.hostname,
            startup_command=container_setup.spec.command,
            hosts_path=hosts_path,
        )

        cmd = [
            self._config.apptainer_executable,
            "instance",
            "start",
            *self._instance_start_flags(container),
            str(container.image_path),
            container.instance_name,
        ]
        result = await self._run(cmd, timeout=60, container_id=container_id, original_cmd=cmd)
        if result.exit_code != 0:
            raise RuntimeError(
                f"Failed to start Apptainer instance {container.instance_name}: "
                f"{result.combined or ''}"
            )

        async with self._batch_state._lock:
            self._batch_state.containers[container_id] = container
            self._batch_state.execution_containers.setdefault(execution_id, set()).add(container_id)

        return container

    async def _clone_container(
        self,
        source_container_id: ContainerID,
        execution_id: str,
        role_name: str,
    ) -> ApptainerContainer:
        if self._batch_state is None:
            raise RuntimeError("clone_sandbox_state called outside of setup_batch context")
        async with self._batch_state._lock:
            source = self._batch_state.containers.get(source_container_id)
        if source is None:
            raise ValueError(f"Cannot clone unknown Apptainer container id: {source_container_id}")

        return await self._start_container_from_source(
            source,
            execution_id=execution_id,
            role_name=role_name,
        )

    async def _start_container_from_source(
        self,
        source: ApptainerContainer,
        execution_id: str,
        role_name: str,
    ) -> ApptainerContainer:
        if self._batch_state is None:
            raise RuntimeError("clone_sandbox_state called outside of setup_batch context")

        container_id = f"{execution_id}:{role_name}"
        instance_name = self._instance_name(execution_id, role_name)
        overlay_path = self._config.work_dir / f"{self._batch_state.batch_id}-{execution_id}-{role_name}.img"
        shutil.copy2(source.overlay_path, overlay_path)
        hosts_path = None
        if source.hosts_path is not None:
            hosts_path = self._hosts_path(execution_id, role_name)
            shutil.copy2(source.hosts_path, hosts_path)

        container = ApptainerContainer(
            container_id=container_id,
            instance_name=instance_name,
            image_path=source.image_path,
            overlay_path=overlay_path,
            environment=dict(source.environment),
            hostname=source.hostname,
            startup_command=source.startup_command,
            hosts_path=hosts_path,
        )
        cmd = [
            self._config.apptainer_executable,
            "instance",
            "start",
            *self._instance_start_flags(container),
            str(container.image_path),
            container.instance_name,
        ]
        result = await self._run(cmd, timeout=60, container_id=container_id, original_cmd=cmd)
        if result.exit_code != 0:
            raise RuntimeError(
                f"Failed to start cloned Apptainer instance {container.instance_name}: "
                f"{result.combined or ''}"
            )

        async with self._batch_state._lock:
            self._batch_state.containers[container_id] = container
            self._batch_state.execution_containers.setdefault(execution_id, set()).add(container_id)
        return container

    async def _start_background_command(self, container: ApptainerContainer) -> None:
        if container.startup_command is None:
            return
        command = self._normalize_shell_command(container.startup_command)
        background = f"nohup {command} >/tmp/prompt-siren-service.log 2>&1 &"
        result = await self.exec(
            container.container_id,
            background,
            timeout=self._config.service_start_timeout,
            shell_path=Path("/bin/sh"),
        )
        if result.exit_code != 0:
            raise RuntimeError(
                f"Failed to start service command in {container.container_id}: "
                f"{result.combined or ''}"
            )

    def _instance_start_flags(self, container: ApptainerContainer) -> list[str]:
        flags: list[str] = ["--overlay", str(container.overlay_path)]
        flags.extend(self._common_flags(include_hostname=container.hostname is not None))
        if container.hosts_path is not None:
            flags.extend(["--bind", f"{container.hosts_path}:/etc/hosts:ro"])
        if container.hostname:
            flags.extend(["--hostname", container.hostname])
        for key, value in container.environment.items():
            flags.extend(["--env", f"{key}={value}"])
        return flags

    def _hosts_path(self, execution_id: str, role_name: str) -> Path:
        if self._batch_state is None:
            raise RuntimeError("hosts file requested outside of setup_batch context")
        return self._config.work_dir / (
            f"{self._batch_state.batch_id}-{execution_id}-{role_name}.hosts"
        )

    def _create_hosts_file(
        self,
        execution_id: str,
        role_name: str,
        host_aliases: Sequence[str],
    ) -> Path | None:
        aliases = tuple(dict.fromkeys(host_aliases))
        if not aliases:
            return None
        for hostname in aliases:
            if not hostname or any(character.isspace() for character in hostname):
                raise ValueError(f"Invalid service hostname: {hostname!r}")

        hosts_path = self._hosts_path(execution_id, role_name)
        content = Path("/etc/hosts").read_text()
        if content and not content.endswith("\n"):
            content += "\n"
        content += "".join(f"127.0.0.1\t{hostname}\n" for hostname in aliases)
        hosts_path.write_text(content)
        return hosts_path

    def _exec_flags(self, extra_env: dict[str, str] | None) -> list[str]:
        flags = self._common_flags(include_hostname=False)
        if extra_env:
            for key, value in extra_env.items():
                flags.extend(["--env", f"{key}={value}"])
        return flags

    def _common_flags(self, include_hostname: bool) -> list[str]:
        flags: list[str] = []
        if self._config.contain:
            flags.append("--contain")
        if self._config.cleanenv:
            flags.append("--cleanenv")
        if self._config.fakeroot:
            flags.append("--fakeroot")
        if self._config.nv:
            flags.append("--nv")
        for bind_path in self._config.bind_paths:
            flags.extend(["--bind", str(bind_path)])
        return flags

    async def _create_overlay(self, overlay_path: Path) -> None:
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            self._config.apptainer_executable,
            "overlay",
            "create",
            "--size",
            str(self._config.overlay_size_mb),
            str(overlay_path),
        ]
        result = await self._run(cmd, timeout=120, container_id=str(overlay_path), original_cmd=cmd)
        if result.exit_code != 0:
            raise RuntimeError(
                f"Failed to create Apptainer overlay {overlay_path}: {result.combined or ''}"
            )

    async def _run(
        self,
        cmd: list[str],
        timeout: int,
        container_id: ContainerID,
        original_cmd: str | list[str],
        stdin: str | bytes | None = None,
    ) -> ExecOutput:
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE if stdin is not None else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdin_bytes = stdin.encode() if isinstance(stdin, str) else stdin
            stdout, stderr = await asyncio.wait_for(
                process.communicate(stdin_bytes), timeout=timeout
            )
        except TimeoutError as e:
            raise ExecTimeoutError(container_id, original_cmd, timeout) from e

        outputs = []
        if stdout:
            outputs.append(StdoutChunk(stdout.decode(errors="replace")))
        if stderr:
            outputs.append(StderrChunk(stderr.decode(errors="replace")))
        return ExecOutput(outputs=outputs, exit_code=process.returncode or 0)

    def _validate_task_setup(self, task_setup: SandboxTaskSetup) -> None:
        setups = [task_setup.agent_container, *task_setup.service_containers.values()]
        for setup in setups:
            if not isinstance(setup.spec.image_spec, PullImageSpec):
                raise NotImplementedError(
                    "ApptainerSandboxManager requires pre-converted SIF images referenced by "
                    "PullImageSpec. BuildImageSpec and DerivedImageSpec are not supported at runtime."
                )
            if setup.spec.ports:
                raise NotImplementedError(
                    "ApptainerSandboxManager does not yet support explicit port mappings"
                )

    def _resolve_image_path(self, container_setup: ContainerSetup) -> Path:
        image_spec = container_setup.spec.image_spec
        if not isinstance(image_spec, PullImageSpec):
            raise NotImplementedError("Apptainer runtime only supports PullImageSpec")

        configured = self._config.image_overrides.get(image_spec.tag)
        image_path = (
            configured
            if configured is not None
            else self._config.image_dir / image_tag_to_sif_name(image_spec.tag)
        )
        image_path = Path(os.path.expandvars(str(image_path))).expanduser()
        if not image_path.exists():
            raise FileNotFoundError(
                f"No SIF image found for tag '{image_spec.tag}'. Expected: {image_path}. "
                "Pre-convert the Docker image to this path, or set "
                "sandbox_manager.config.image_overrides for this tag."
            )
        return image_path

    async def _cleanup_container(self, container_id: ContainerID) -> None:
        if self._batch_state is None:
            return
        async with self._batch_state._lock:
            container = self._batch_state.containers.pop(container_id, None)
            if container is not None:
                execution_id = container_id.split(":", 1)[0]
                container_set = self._batch_state.execution_containers.get(execution_id)
                if container_set is not None:
                    container_set.discard(container_id)
                    if not container_set:
                        self._batch_state.execution_containers.pop(execution_id, None)
        if container is None:
            return

        await self._run(
            [self._config.apptainer_executable, "instance", "stop", container.instance_name],
            timeout=30,
            container_id=container_id,
            original_cmd="instance stop",
        )
        try:
            container.overlay_path.unlink(missing_ok=True)
        except OSError:
            pass
        if container.hosts_path is not None:
            try:
                container.hosts_path.unlink(missing_ok=True)
            except OSError:
                pass

    async def _cleanup_batch(self) -> None:
        if self._batch_state is None:
            return
        async with self._batch_state._lock:
            container_ids = list(self._batch_state.containers)
        await asyncio.gather(
            *(self._cleanup_container(container_id) for container_id in container_ids),
            return_exceptions=True,
        )

    def _instance_name(self, execution_id: str, role_name: str) -> str:
        if self._batch_state is None:
            raise RuntimeError("Apptainer batch state is not initialized")
        safe_role = re.sub(r"[^A-Za-z0-9_.-]+", "_", role_name)
        return f"{self._batch_state.batch_id}-{execution_id[:12]}-{safe_role}"

    @staticmethod
    def _normalize_shell_command(cmd: str | list[str]) -> str:
        if isinstance(cmd, list):
            return " ".join(shlex.quote(arg) for arg in cmd)
        return cmd


def create_apptainer_sandbox_manager(
    config: ApptainerSandboxConfig, context: None = None
) -> ApptainerSandboxManager:
    """Factory function to create an Apptainer sandbox manager."""
    return ApptainerSandboxManager(config)
