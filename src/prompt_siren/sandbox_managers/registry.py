# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Registry for sandbox manager types and their factory functions.

This module provides a centralized registry for sandbox manager types, allowing for
dynamic sandbox manager creation with type safety.
"""

from collections.abc import Callable
from typing import TypeAlias, TypeVar

from pydantic import BaseModel

from ..registry_base import BaseRegistry
from .abstract import AbstractSandboxManager

ConfigT = TypeVar("ConfigT", bound=BaseModel)

# Type alias for sandbox manager factory functions
SandboxManagerFactory: TypeAlias = Callable[[ConfigT], AbstractSandboxManager]


# Create a global sandbox manager registry instance
sandbox_registry = BaseRegistry[AbstractSandboxManager, None](
    "sandbox_manager", "prompt_siren.sandbox_managers"
)


# Convenience functions for sandbox manager-specific naming
def register_sandbox_manager(
    sandbox_type: str,
    config_class: type[ConfigT],
    factory: SandboxManagerFactory[ConfigT],
) -> None:
    """Register a sandbox manager type with its configuration class and factory."""
    sandbox_registry.register(sandbox_type, config_class, factory)


def get_sandbox_config_class(sandbox_type: str) -> type[BaseModel]:
    """Get the configuration class for a sandbox manager type."""
    _register_builtin_fallback(sandbox_type)
    config_class = sandbox_registry.get_config_class(sandbox_type)
    if config_class is None:
        raise RuntimeError(f"Sandbox manager type '{sandbox_type}' must have a config class")
    return config_class


def create_sandbox_manager(sandbox_type: str, config: BaseModel) -> AbstractSandboxManager:
    """Create a sandbox manager instance from a configuration."""
    _register_builtin_fallback(sandbox_type)
    return sandbox_registry.create_component(sandbox_type, config)


def get_registered_sandbox_managers() -> list[str]:
    """Get a list of all registered sandbox manager types."""
    _register_builtin_fallback("apptainer")
    return sandbox_registry.get_registered_components()


def _register_builtin_fallback(sandbox_type: str) -> None:
    """Register built-in managers when editable metadata is stale.

    Entry point metadata is normally the source of truth. On clusters, users may
    transfer source changes without reinstalling the package, so this fallback
    keeps built-in managers available from the source tree.
    """
    if sandbox_type != "apptainer":
        return
    if sandbox_type in sandbox_registry._registry:
        return
    from .apptainer import ApptainerSandboxConfig, create_apptainer_sandbox_manager

    sandbox_registry.register(
        "apptainer",
        ApptainerSandboxConfig,
        create_apptainer_sandbox_manager,
    )
