"""Kairos Agent — Versioned Prompt Registry.

Loads prompt templates from ``.txt`` files and extracts YAML front-matter
metadata (version, author, description).  Provides ``RenderedPrompt``
objects that carry the rendered text alongside version provenance so callers
can record *which* prompt version produced each LLM response.

Convention — prompt files support an optional YAML front-matter header::

    ---
    version: 2
    description: "Hook caption writer — Zeigarnik framing"
    ---
    You are a short-form video caption writer …

If the header is absent the template is assigned ``version=0`` (unversioned).

Finding 3.2: *"No Prompt Versioning or Central Management"*
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Match optional YAML front-matter: ``---\n<yaml>\n---\n<body>``
_FRONT_MATTER_RE = re.compile(
    r"\A---\s*\n(?P<meta>.*?)\n---\s*\n(?P<body>.*)\Z",
    re.DOTALL,
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PromptMeta:
    """Metadata extracted from a prompt template's front-matter."""

    version: int = 0
    description: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    """A prompt rendered from a versioned template."""

    text: str
    template_name: str
    version: int
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise for storage in step artifacts / AgentRun."""
        return {
            "template": self.template_name,
            "version": self.version,
            "description": self.description,
            "char_count": len(self.text),
        }


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_front_matter(raw: str) -> tuple[PromptMeta, str]:
    """Split *raw* into ``(PromptMeta, body_text)``.

    If no front-matter block is found the entire string is treated as body
    and metadata defaults to ``version=0``.
    """
    m = _FRONT_MATTER_RE.match(raw)
    if not m:
        return PromptMeta(), raw.strip()

    body = m.group("body").strip()
    meta_block = m.group("meta")

    # Lightweight YAML parsing (avoids PyYAML dependency).
    # We only need simple ``key: value`` lines.
    attrs: dict[str, Any] = {}
    for line in meta_block.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        attrs[key] = value

    version = int(attrs.pop("version", 0))
    description = str(attrs.pop("description", ""))
    return PromptMeta(version=version, description=description, extra=attrs), body


def _render_template(template: str, variables: dict[str, str]) -> str:
    """Replace ``{{ key }}`` placeholders with values from *variables*."""
    result = template
    for key, value in variables.items():
        result = result.replace("{{ " + key + " }}", str(value))
        result = result.replace("{{" + key + "}}", str(value))
    return result


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class PromptRegistry:
    """Load, cache, and render versioned prompt templates.

    Usage::

        registry = PromptRegistry(base_dir)
        rp = registry.render("system/concept_developer", {"category": "ball_pit"})
        print(rp.version, rp.text[:80])
    """

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        self._cache: dict[str, tuple[PromptMeta, str]] = {}

    # -- internal ----------------------------------------------------------

    def _load(self, relative_path: str) -> tuple[PromptMeta, str]:
        """Load and parse a template file (cached after first read)."""
        if relative_path in self._cache:
            return self._cache[relative_path]

        path = self._base_dir / relative_path
        if not path.exists():
            raise FileNotFoundError(f"Prompt template not found: {path}")

        raw = path.read_text(encoding="utf-8")
        meta, body = _parse_front_matter(raw)
        self._cache[relative_path] = (meta, body)

        if meta.version == 0:
            logger.debug(
                "Prompt '%s' has no version front-matter (treated as v0)",
                relative_path,
            )
        return meta, body

    # -- public API --------------------------------------------------------

    def render(
        self,
        template_name: str,
        variables: dict[str, str] | None = None,
    ) -> RenderedPrompt:
        """Render a prompt template with optional variable substitution.

        Args:
            template_name: Relative path inside ``base_dir`` (without
                leading slash).  E.g. ``"system/caption_writer.txt"``.
            variables: ``{{ key }}`` replacement values.

        Returns:
            ``RenderedPrompt`` with the rendered text *and* version metadata.
        """
        meta, body = self._load(template_name)
        text = _render_template(body, variables or {})
        return RenderedPrompt(
            text=text,
            template_name=template_name,
            version=meta.version,
            description=meta.description,
        )

    def get_version(self, template_name: str) -> int:
        """Return the version number of a template without rendering."""
        meta, _ = self._load(template_name)
        return meta.version

    def list_templates(self) -> list[dict[str, Any]]:
        """List all ``.txt`` templates under the base directory."""
        results: list[dict[str, Any]] = []
        for path in sorted(self._base_dir.rglob("*.txt")):
            rel = str(path.relative_to(self._base_dir)).replace("\\", "/")
            meta, body = self._load(rel)
            results.append({
                "template": rel,
                "version": meta.version,
                "description": meta.description,
                "chars": len(body),
            })
        return results
