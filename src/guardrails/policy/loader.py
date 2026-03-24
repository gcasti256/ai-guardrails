"""YAML-based policy loader and validator.

Loads :class:`Policy` definitions from YAML files, strings, or entire
directories and performs structural validation beyond what Pydantic
enforces (e.g. rule-chain references, duplicate names).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from guardrails.policy.models import DetectorType, Policy, PolicyRule, RuleAction

logger = logging.getLogger(__name__)


class PolicyLoadError(Exception):
    """Raised when a policy file cannot be loaded or parsed."""


class PolicyLoader:
    """Load, validate, and merge YAML policy definitions.

    Example usage::

        loader = PolicyLoader()
        policy = loader.load_file("policies/default.yaml")
        errors = loader.validate(policy)
        if errors:
            raise ValueError(errors)
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_file(self, path: str | Path) -> Policy:
        """Load a single policy from a YAML file.

        Args:
            path: Filesystem path to a ``.yaml`` / ``.yml`` file.

        Returns:
            Parsed and validated :class:`Policy`.

        Raises:
            PolicyLoadError: If the file cannot be read or contains
                invalid YAML / policy structure.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise PolicyLoadError(f"Policy file not found: {file_path}")
        if not file_path.is_file():
            raise PolicyLoadError(f"Path is not a file: {file_path}")

        try:
            raw = file_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise PolicyLoadError(f"Cannot read {file_path}: {exc}") from exc

        return self.load_string(raw, source=str(file_path))

    def load_string(self, raw: str, *, source: str = "<string>") -> Policy:
        """Parse a policy from a raw YAML string.

        Args:
            raw: YAML content.
            source: Label used in error messages to identify the origin.

        Returns:
            Parsed :class:`Policy`.

        Raises:
            PolicyLoadError: On YAML syntax errors or validation failures.
        """
        try:
            data = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise PolicyLoadError(f"YAML parse error in {source}: {exc}") from exc

        if not isinstance(data, dict):
            raise PolicyLoadError(
                f"Expected a YAML mapping at the top level in {source}, "
                f"got {type(data).__name__}"
            )

        return self._parse_policy(data, source=source)

    def load_directory(self, path: str | Path) -> list[Policy]:
        """Load every ``.yaml`` / ``.yml`` file in *path* as a policy.

        Files that fail to load are logged as warnings and skipped so
        that one bad file does not prevent the rest from loading.

        Args:
            path: Directory containing policy YAML files.

        Returns:
            List of successfully loaded policies (may be empty).

        Raises:
            PolicyLoadError: If *path* is not a directory.
        """
        dir_path = Path(path)
        if not dir_path.is_dir():
            raise PolicyLoadError(f"Not a directory: {dir_path}")

        policies: list[Policy] = []
        yaml_files = sorted(
            p for p in dir_path.iterdir() if p.suffix in {".yaml", ".yml"} and p.is_file()
        )

        for file_path in yaml_files:
            try:
                policy = self.load_file(file_path)
                policies.append(policy)
                logger.info("Loaded policy %r from %s", policy.name, file_path)
            except PolicyLoadError as exc:
                logger.warning("Skipping %s: %s", file_path, exc)

        return policies

    def validate(self, policy: Policy) -> list[str]:
        """Run additional structural checks on *policy*.

        Returns a (possibly empty) list of human-readable error strings.
        Pydantic already enforces type/shape constraints; this method
        catches higher-level issues such as:

        * Duplicate rule names.
        * Chain references to non-existent rules.
        * Disabled rules referenced in chains.
        * Rules with unknown detector types (future-proofing hint).
        """
        errors: list[str] = []

        # --- Duplicate rule names ---
        seen_names: dict[str, int] = {}
        for rule in policy.rules:
            seen_names[rule.name] = seen_names.get(rule.name, 0) + 1
        for name, count in seen_names.items():
            if count > 1:
                errors.append(f"Duplicate rule name '{name}' appears {count} times.")

        # --- Chain reference checks ---
        rule_names = {r.name for r in policy.rules}
        enabled_names = {r.name for r in policy.rules if r.enabled}

        for rule in policy.rules:
            for chained in rule.chain:
                if chained not in rule_names:
                    errors.append(
                        f"Rule '{rule.name}' chains to '{chained}' which does not exist."
                    )
                elif chained not in enabled_names:
                    errors.append(
                        f"Rule '{rule.name}' chains to '{chained}' which is disabled."
                    )
                if chained == rule.name:
                    errors.append(
                        f"Rule '{rule.name}' chains to itself (circular reference)."
                    )

        # --- No enabled rules warning ---
        if policy.rules and not any(r.enabled for r in policy.rules):
            errors.append("Policy has rules but none are enabled.")

        return errors

    @staticmethod
    def merge(policies: list[Policy], *, name: str = "merged") -> Policy:
        """Merge multiple policies into one by combining their rules.

        Rules are collected in order; if two policies define a rule with
        the same name the later one wins.  The merged policy inherits
        the ``default_action`` from the *first* policy.

        Args:
            policies: Policies to merge.
            name: Name for the resulting policy.

        Returns:
            A new :class:`Policy` containing all unique rules.
        """
        if not policies:
            return Policy(name=name)

        rules_by_name: dict[str, PolicyRule] = {}
        for policy in policies:
            for rule in policy.rules:
                rules_by_name[rule.name] = rule

        merged_metadata: dict[str, Any] = {}
        for policy in policies:
            merged_metadata.update(policy.metadata)

        return Policy(
            name=name,
            version=policies[0].version,
            description=f"Merged from: {', '.join(p.name for p in policies)}",
            rules=list(rules_by_name.values()),
            default_action=policies[0].default_action,
            metadata=merged_metadata,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_policy(self, data: dict[str, Any], *, source: str) -> Policy:
        """Convert a raw dict into a validated :class:`Policy`."""
        try:
            return Policy.model_validate(data)
        except ValidationError as exc:
            messages = "; ".join(
                f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}"
                for e in exc.errors()
            )
            raise PolicyLoadError(
                f"Validation failed for policy in {source}: {messages}"
            ) from exc
