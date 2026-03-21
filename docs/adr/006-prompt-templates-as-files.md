# ADR-006: Prompt Templates as Versioned Files

**Status:** Accepted
**Date:** 2026-02-18

## Context

Prompts are the primary interface between our code and LLM intelligence. They need to be:
- Version-controlled (track changes over time)
- Reviewable (PR diffs should show prompt changes clearly)
- Testable (render with test data, validate output format)
- Structured (consistent system/user split)
- Parameterized (inject pipeline-specific variables)

Options considered:
1. **Inline strings** — `f"You are a ..."` in Python code. Simple but hard to review, no separation.
2. **Prompt management platform** — PromptLayer, Langfuse prompt management. External dependency.
3. **Versioned files** — Jinja2 `.txt` templates in the repo, rendered by builder modules.

## Decision

Prompts are Jinja2 `.txt` files organized by pipeline and role:

```
ai/prompts/
├── physics/
│   ├── system/concept_developer.txt
│   ├── user/concept_developer.txt
│   ├── system/simulation_config.txt
│   └── user/simulation_config.txt
├── domino/
│   ├── system/concept_developer.txt
│   └── user/concept_developer.txt
└── shared/
    └── ...
```

Each pipeline has a `builder.py` module with typed render functions:

```python
def render_concept_user(
    inventory: InventoryReport,
    category: str,
    knowledge: str,
) -> str:
    template = _env.get_template("physics/user/concept_developer.txt")
    return template.render(inventory=inventory, category=category, knowledge=knowledge)
```

Structured output schemas are provided by Instructor's `response_model` parameter — not injected into the prompt text.

## Consequences

**Positive:**
- Prompt changes show up as clear diffs in git history and PRs.
- Templates are testable — `test_prompt_harness.py` verifies all categories have templates and they render without errors.
- Builder functions enforce which variables are available (type-checked, no undefined template vars).
- Separation of concerns — prompt text in `.txt`, rendering logic in Python.

**Negative:**
- Two files to edit for a prompt change (template + builder if variables change).
- Jinja2 syntax is one more thing to learn (though it's minimal — mostly `{{ var }}` substitution).
- No prompt versioning metadata (version number, author, date). Use git blame instead.

**Schema deduplication (Phase 4):** Originally, JSON schemas were manually injected into prompts via `model_json_schema()`. This was removed in Phase 4 because Instructor already supplies the schema to the LLM via function calling / tool use. Templates now say "Output ONLY valid JSON matching the response schema" without embedding the schema text.
