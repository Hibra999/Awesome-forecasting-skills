#!/usr/bin/env python3
"""Validate the forecasting skills repository contract."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - depends on local environment.
    raise SystemExit(
        "validate_skill_repo.py requires PyYAML. Install it with: python -m pip install pyyaml"
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "skills.catalog.yaml"
README = ROOT / "README.md"
REQUIRED_AGENT_FIELDS = ("display_name", "short_description", "default_prompt")


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def load_yaml(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_frontmatter(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n", text, flags=re.DOTALL)
    if not match:
        raise ValueError("missing YAML frontmatter")
    data = yaml.safe_load(match.group(1))
    if not isinstance(data, dict):
        raise ValueError("frontmatter is not a mapping")
    return data


def read_readme_skill_ids() -> set[str]:
    if not README.exists():
        return set()
    ids: set[str] = set()
    for line in README.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\|\s*`([^`]+)`\s*\|", line)
        if match:
            ids.add(match.group(1))
    return ids


def main() -> int:
    errors: list[str] = []

    if not CATALOG.exists():
        print("ERROR: missing skills.catalog.yaml")
        return 1

    catalog = load_yaml(CATALOG)
    if not isinstance(catalog, dict) or catalog.get("schema_version") != 1:
        errors.append("skills.catalog.yaml must declare schema_version: 1")

    skills = catalog.get("skills") if isinstance(catalog, dict) else None
    if not isinstance(skills, list) or not skills:
        errors.append("skills.catalog.yaml must contain a non-empty skills list")
        skills = []

    ids: set[str] = set()
    catalog_paths: set[str] = set()

    for index, skill in enumerate(skills):
        if not isinstance(skill, dict):
            errors.append(f"skills[{index}] must be a mapping")
            continue

        skill_id = skill.get("id")
        path_value = skill.get("path")
        if not isinstance(skill_id, str) or not skill_id:
            errors.append(f"skills[{index}] is missing id")
            continue
        if skill_id in ids:
            errors.append(f"{skill_id}: duplicate id")
        ids.add(skill_id)

        for field in ("domain", "stage", "path", "depends_on", "agent_config", "references", "scripts"):
            if field not in skill:
                errors.append(f"{skill_id}: missing {field}")

        if not isinstance(path_value, str) or path_value.startswith("/") or ".." in Path(path_value).parts:
            errors.append(f"{skill_id}: path must be a safe relative path")
            continue

        skill_dir = ROOT / path_value
        skill_md = skill_dir / "SKILL.md"
        catalog_paths.add(path_value)

        if not skill_md.exists():
            errors.append(f"{skill_id}: missing {rel(skill_md)}")
            continue

        try:
            frontmatter = read_frontmatter(skill_md)
        except ValueError as exc:
            errors.append(f"{skill_id}: {exc}")
            frontmatter = {}

        if frontmatter.get("name") != skill_id:
            errors.append(f"{skill_id}: SKILL.md frontmatter name must match catalog id")
        if not frontmatter.get("description"):
            errors.append(f"{skill_id}: SKILL.md frontmatter missing description")

        text = skill_md.read_text(encoding="utf-8").lower()
        if "leak" not in text:
            errors.append(f"{skill_id}: SKILL.md must mention leakage safeguards")
        if "valid" not in text:
            errors.append(f"{skill_id}: SKILL.md must mention validation")

        agent_config = skill.get("agent_config")
        if isinstance(agent_config, str):
            agent_path = ROOT / agent_config
            if not agent_path.exists():
                errors.append(f"{skill_id}: missing {agent_config}")
            else:
                agent = load_yaml(agent_path)
                interface = agent.get("interface") if isinstance(agent, dict) else None
                if not isinstance(interface, dict):
                    errors.append(f"{skill_id}: agent config missing interface")
                else:
                    for field in REQUIRED_AGENT_FIELDS:
                        if not interface.get(field):
                            errors.append(f"{skill_id}: agent config missing interface.{field}")

        references = skill.get("references")
        if isinstance(references, str):
            ref_path = ROOT / references
            if not ref_path.is_dir():
                errors.append(f"{skill_id}: missing references directory {references}")
            elif not any(ref_path.glob("*.md")):
                errors.append(f"{skill_id}: references directory has no markdown files")

        scripts = skill.get("scripts")
        if not isinstance(scripts, list):
            errors.append(f"{skill_id}: scripts must be a list")
        else:
            for script in scripts:
                if not isinstance(script, str) or not (ROOT / script).is_file():
                    errors.append(f"{skill_id}: declared script does not exist: {script}")

    for skill in skills:
        if not isinstance(skill, dict) or not isinstance(skill.get("id"), str):
            continue
        skill_id = skill["id"]
        depends_on = skill.get("depends_on")
        if not isinstance(depends_on, list):
            errors.append(f"{skill_id}: depends_on must be a list")
            continue
        for dependency in depends_on:
            if dependency == skill_id:
                errors.append(f"{skill_id}: cannot depend on itself")
            if dependency not in ids:
                errors.append(f"{skill_id}: unknown dependency {dependency}")

    actual_paths = {rel(path.parent) for path in ROOT.glob("*/SKILL.md")}
    missing_from_catalog = sorted(actual_paths - catalog_paths)
    stale_catalog_paths = sorted(catalog_paths - actual_paths)
    if missing_from_catalog:
        errors.append(f"skills missing from catalog: {', '.join(missing_from_catalog)}")
    if stale_catalog_paths:
        errors.append(f"catalog paths without SKILL.md: {', '.join(stale_catalog_paths)}")

    readme_ids = read_readme_skill_ids()
    if readme_ids:
        missing_from_readme = sorted(ids - readme_ids)
        stale_readme_ids = sorted(readme_ids - ids)
        if missing_from_readme:
            errors.append(f"catalog skills missing from README table: {', '.join(missing_from_readme)}")
        if stale_readme_ids:
            errors.append(f"README skills missing from catalog: {', '.join(stale_readme_ids)}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print(f"OK: {len(ids)} skills validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
