"""Build local SWE-bench images for one Prompt Siren benign/attack pair.

This is useful when the default registry images are for the wrong architecture.
Run experiments with `dataset.config.registry=null` after building these images.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from prompt_siren.build_images import ImageBuilder
from prompt_siren.datasets import get_dataset_config_class, get_image_build_specs
from prompt_siren.sandbox_managers.docker.manager import create_docker_client_from_config
from prompt_siren.sandbox_managers.image_spec import DerivedImageSpec, MultiStageBuildImageSpec


def _tag_of(spec) -> str:
    if isinstance(spec, MultiStageBuildImageSpec):
        return spec.stages[-1].tag
    return spec.tag


def _matches_pair(spec, instance_id: str, attack_id: str) -> bool:
    tag = _tag_of(spec)
    wanted_suffixes = {
        f"siren-swebench-benign:{instance_id}",
        f"siren-swebench-service:{attack_id}",
        f"siren-swebench-pair:{instance_id}__{attack_id}",
    }
    return any(tag.endswith(suffix) for suffix in wanted_suffixes)


async def _build(args: argparse.Namespace) -> None:
    config_class = get_dataset_config_class("swebench")
    config = config_class(
        dataset_name=args.dataset_name,
        enable_source_injection=args.enable_source_injection,
        enable_skill_injection=args.enable_skill_injection,
    )

    specs = get_image_build_specs("swebench", config, args.cache_dir)
    selected_specs = [
        spec
        for spec in specs
        if _matches_pair(spec, args.instance_id, args.attack_id)
    ]

    if not selected_specs:
        raise SystemExit(
            "No image specs matched. Check the instance id and attack id."
        )

    print("Building local images:")
    for spec in selected_specs:
        kind = "derived" if isinstance(spec, DerivedImageSpec) else "base"
        print(f"  [{kind}] {_tag_of(spec)}")

    docker = create_docker_client_from_config("local", {})
    try:
        builder = ImageBuilder(
            docker_client=docker,
            rebuild_existing=args.rebuild_existing,
            registry=None,
        )
        errors = await builder.build_all_specs(selected_specs)
    finally:
        await docker.close()

    if errors:
        print("\nBuild failed:")
        for error in errors:
            print(f"  {error.image_tag}: {type(error.error).__name__}: {error.error}")
        raise SystemExit(1)

    print("\nDone. Run prompt-siren with: dataset.config.registry=null")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build local Docker images for one Prompt Siren SWE-bench task pair."
    )
    parser.add_argument(
        "--instance-id",
        default="astropy__astropy-12907",
        help="SWE-bench benign instance id.",
    )
    parser.add_argument(
        "--attack-id",
        default="env_direct_exfil_task",
        help="Prompt Siren SWE-bench malicious task id.",
    )
    parser.add_argument(
        "--dataset-name",
        default="SWE-bench/SWE-bench_Lite",
        help="SWE-bench dataset name or local dataset path.",
    )
    parser.add_argument(
        "--cache-dir",
        default=".siren-docker-cache/swebench",
        help="Build context cache directory.",
    )
    parser.add_argument(
        "--enable-source-injection",
        dest="enable_source_injection",
        action="store_true",
        default=False,
        help="Insert the injection into the mapped source-code location.",
    )
    parser.add_argument(
        "--disable-source-injection",
        dest="enable_source_injection",
        action="store_false",
        help="Do not insert the injection into the mapped source-code location.",
    )
    parser.add_argument(
        "--enable-skill-injection",
        dest="enable_skill_injection",
        action="store_true",
        default=True,
        help="Create repository SKILL.md files containing the injection.",
    )
    parser.add_argument(
        "--disable-skill-injection",
        dest="enable_skill_injection",
        action="store_false",
        help="Do not create repository SKILL.md files containing the injection.",
    )
    parser.add_argument(
        "--rebuild-existing",
        action="store_true",
        default=True,
        help="Delete and rebuild matching images if they already exist.",
    )
    parser.add_argument(
        "--no-rebuild-existing",
        dest="rebuild_existing",
        action="store_false",
        help="Skip matching images that already exist locally.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose build logs.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    asyncio.run(_build(args))


if __name__ == "__main__":
    main()
