# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Apptainer-based sandbox manager implementation."""

from .manager import (
    ApptainerSandboxConfig,
    ApptainerSandboxManager,
    create_apptainer_sandbox_manager,
    image_tag_to_sif_name,
)

__all__ = [
    "ApptainerSandboxConfig",
    "ApptainerSandboxManager",
    "create_apptainer_sandbox_manager",
    "image_tag_to_sif_name",
]
