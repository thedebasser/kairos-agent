"""Kairos Agent — Prompt Development Harness.

Standalone CLI tool for manually iterating on simulation prompts.
Used during development to refine prompt templates and build the
initial "golden set" of working simulations.

Usage:
    python -m kairos.tools.prompt_harness --category ball_pit
    python -m kairos.tools.prompt_harness --category marble_funnel --seed 123
    python -m kairos.tools.prompt_harness --list-categories
    python -m kairos.tools.prompt_harness --dry-run --category destruction
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from uuid import uuid4

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from kairos.config import get_settings
from kairos.schemas.contracts import (
    AudioBrief,
    ConceptBrief,
    EnergyLevel,
    PipelineState,
    ScenarioCategory,
    SimulationRequirements,
)
from kairos.pipelines.physics.simulation_agent import PhysicsSimulationAgent

logger = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Sample concepts for each category (used when no concept JSON is provided)
# ---------------------------------------------------------------------------

SAMPLE_CONCEPTS: dict[str, ConceptBrief] = {
    "ball_pit": ConceptBrief(
        pipeline="physics",
        category=ScenarioCategory.BALL_PIT,
        title="Rainbow Ball Pit Overflow",
        visual_brief=(
            "Hundreds of colourful balls drop into a tall glass container, "
            "bouncing and piling up until the container overflows in a "
            "satisfying cascade of colour."
        ),
        simulation_requirements=SimulationRequirements(
            body_count_initial=5,
            body_count_max=300,
            interaction_type="bounce_accumulate",
            colour_palette=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"],
            background_colour="#1a1a2e",
            special_effects=[],
        ),
        audio_brief=AudioBrief(
            mood=["upbeat", "playful"],
            tempo_bpm_min=100,
            tempo_bpm_max=130,
            energy_curve=EnergyLevel.BUILDING,
        ),
        hook_text="Watch it overflow",
        novelty_score=7.0,
        feasibility_score=8.5,
        target_duration_sec=65,
        seed=42,
    ),
    "marble_funnel": ConceptBrief(
        pipeline="physics",
        category=ScenarioCategory.MARBLE_FUNNEL,
        title="Spiral Marble Run",
        visual_brief=(
            "Marbles are dropped from the top of a spiral funnel track, "
            "accelerating as they descend through increasingly tight curves "
            "before collecting in a mesmerizing swirl at the bottom."
        ),
        simulation_requirements=SimulationRequirements(
            body_count_initial=3,
            body_count_max=200,
            interaction_type="roll_collect",
            colour_palette=["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6"],
            background_colour="#0d1117",
            special_effects=["trails"],
        ),
        audio_brief=AudioBrief(
            mood=["mesmerizing", "flowing"],
            tempo_bpm_min=90,
            tempo_bpm_max=120,
            energy_curve=EnergyLevel.BUILDING,
        ),
        hook_text="So satisfying to watch",
        novelty_score=6.5,
        feasibility_score=7.5,
        target_duration_sec=65,
        seed=123,
    ),
    "domino_chain": ConceptBrief(
        pipeline="physics",
        category=ScenarioCategory.DOMINO_CHAIN,
        title="Colour-Coded Domino Spiral",
        visual_brief=(
            "A massive spiral of colour-coded dominoes is triggered from "
            "the centre, creating an expanding wave of falling dominoes "
            "that reveals a rainbow pattern as they topple outward."
        ),
        simulation_requirements=SimulationRequirements(
            body_count_initial=200,
            body_count_max=500,
            interaction_type="topple_cascade",
            colour_palette=["#FF6B6B", "#FFD93D", "#6BCB77", "#4D96FF", "#9B59B6"],
            background_colour="#16213e",
            special_effects=[],
        ),
        audio_brief=AudioBrief(
            mood=["dramatic", "building"],
            tempo_bpm_min=100,
            tempo_bpm_max=140,
            energy_curve=EnergyLevel.CLIMAX,
        ),
        hook_text="Wait for the spiral",
        novelty_score=7.5,
        feasibility_score=6.0,
        target_duration_sec=65,
        seed=256,
    ),
    "destruction": ConceptBrief(
        pipeline="physics",
        category=ScenarioCategory.DESTRUCTION,
        title="Tower of Blocks Wrecking Ball",
        visual_brief=(
            "A carefully stacked tower of colourful blocks stands tall. "
            "A heavy wrecking ball swings in and smashes through the tower, "
            "sending debris flying in a spectacular explosion of colour."
        ),
        simulation_requirements=SimulationRequirements(
            body_count_initial=150,
            body_count_max=400,
            interaction_type="impact_scatter",
            colour_palette=["#FF4757", "#2ED573", "#1E90FF", "#FFA502", "#A55EEA"],
            background_colour="#1a1a2e",
            special_effects=["particles"],
        ),
        audio_brief=AudioBrief(
            mood=["intense", "explosive"],
            tempo_bpm_min=110,
            tempo_bpm_max=140,
            energy_curve=EnergyLevel.CLIMAX,
        ),
        hook_text="This tower is doomed",
        novelty_score=8.0,
        feasibility_score=7.0,
        target_duration_sec=65,
        seed=512,
    ),
}


# ---------------------------------------------------------------------------
# Harness workflow
# ---------------------------------------------------------------------------


async def run_harness(
    category: str,
    *,
    seed: int | None = None,
    concept_json: str | None = None,
    dry_run: bool = False,
    skip_validation: bool = False,
    max_iterations: int = 1,
) -> None:
    """Run the prompt development harness for a single category.

    Steps:
    1. Build or load a ConceptBrief
    2. Show the generated prompt (for review)
    3. Generate simulation code via LLM
    4. Optionally execute in sandbox
    5. Optionally validate output
    6. Display results
    """
    agent = PhysicsSimulationAgent()

    # ----- Build concept -----
    if concept_json:
        concept = ConceptBrief.model_validate_json(concept_json)
    else:
        if category not in SAMPLE_CONCEPTS:
            console.print(f"[red]Unknown category: {category}[/red]")
            console.print(f"Available: {', '.join(SAMPLE_CONCEPTS)}")
            return
        concept = SAMPLE_CONCEPTS[category]
        if seed is not None:
            # Rebuild with custom seed (ConceptBrief is frozen)
            concept = concept.model_copy(update={"seed": seed})

    state = PipelineState(pipeline="physics")

    # ----- Display concept -----
    console.print(Panel(
        f"[bold]{concept.title}[/bold]\n"
        f"Category: {concept.category.value}\n"
        f"Duration: {concept.target_duration_sec}s | "
        f"Bodies: {concept.simulation_requirements.body_count_initial} -> "
        f"{concept.simulation_requirements.body_count_max}\n"
        f"Seed: {concept.seed}",
        title="Concept Brief",
        border_style="blue",
    ))

    # ----- Show prompt -----
    prompt = agent._build_generation_prompt(concept)  # noqa: SLF001
    console.print(Panel(
        prompt[:2000] + ("..." if len(prompt) > 2000 else ""),
        title="Generated Prompt (first 2000 chars)",
        border_style="cyan",
    ))

    if dry_run:
        console.print("[yellow]Dry run — skipping LLM call and execution[/yellow]")
        return

    # ----- Generate code -----
    console.print("\n[bold]Generating simulation code...[/bold]")
    start = time.monotonic()
    code = await agent.generate_simulation(concept)
    gen_time = time.monotonic() - start
    console.print(f"[green]Generated {len(code)} chars in {gen_time:.1f}s[/green]")

    # Save generated code
    output_dir = Path("output") / "harness" / category
    output_dir.mkdir(parents=True, exist_ok=True)
    code_path = output_dir / f"simulation_{concept.seed}.py"
    code_path.write_text(code, encoding="utf-8")
    console.print(f"Saved code to: {code_path}")

    # ----- Execute -----
    console.print("\n[bold]Executing in sandbox...[/bold]")
    try:
        result = await agent.execute_simulation(code)
    except (SimulationTimeoutError, SimulationOOMError, SimulationExecutionError) as exc:
        console.print(f"[red]Execution failed: {exc}[/red]")
        return

    # Display execution result
    _display_execution_result(result)

    # ----- Find video -----
    video_path = agent._find_video(result)  # noqa: SLF001
    if video_path is None:
        console.print("[red]No MP4 file found in output[/red]")
        return

    console.print(f"\n[green]Video: {video_path}[/green]")

    # ----- Stats -----
    stats = await agent.get_simulation_stats(video_path)
    stats = agent._enrich_stats_from_stdout(stats, result.stdout)  # noqa: SLF001
    _display_stats(stats)

    # ----- Validate -----
    if not skip_validation:
        console.print("\n[bold]Running validation...[/bold]")
        vresult = await agent.validate_output(video_path)
        _display_validation(vresult)

        # ----- Adjustment loop -----
        current_code = code
        for i in range(2, max_iterations + 1):
            if vresult.passed:
                break
            console.print(f"\n[bold yellow]Adjusting (iteration {i})...[/bold yellow]")
            current_code = await agent.adjust_parameters(current_code, vresult, i)
            adj_path = output_dir / f"simulation_{concept.seed}_iter{i}.py"
            adj_path.write_text(current_code, encoding="utf-8")

            try:
                result = await agent.execute_simulation(current_code)
            except (SimulationTimeoutError, SimulationOOMError, SimulationExecutionError) as exc:
                console.print(f"[red]Execution failed (iter {i}): {exc}[/red]")
                continue

            video_path = agent._find_video(result)  # noqa: SLF001
            if video_path is None:
                console.print(f"[red]No MP4 output (iter {i})[/red]")
                continue

            vresult = await agent.validate_output(video_path)
            _display_validation(vresult)

    console.print("\n[bold green]Harness run complete[/bold green]")


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _display_execution_result(result: SimulationResult) -> None:
    """Print execution result details."""
    status = "[green]SUCCESS[/green]" if result.returncode == 0 else "[red]FAILED[/red]"
    console.print(f"Status: {status} (exit code {result.returncode})")
    if result.stderr:
        console.print(Panel(
            result.stderr[:1000],
            title="stderr (first 1000 chars)",
            border_style="red",
        ))
    if result.output_files:
        console.print(f"Output files: {', '.join(result.output_files)}")


def _display_stats(stats: SimulationStats) -> None:
    """Print simulation statistics in a table."""
    table = Table(title="Simulation Stats")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Duration", f"{stats.duration_sec:.1f}s")
    table.add_row("Peak Bodies", str(stats.peak_body_count))
    table.add_row("Avg FPS", f"{stats.avg_fps:.1f}")
    table.add_row("Min FPS", f"{stats.min_fps:.1f}")
    table.add_row("Payoff @", f"{stats.payoff_timestamp_sec:.1f}s")
    table.add_row("Total Frames", str(stats.total_frames))
    table.add_row("File Size", f"{stats.file_size_bytes / 1_000_000:.1f} MB")
    console.print(table)


def _display_validation(vresult: ValidationResult) -> None:
    """Print validation results in a table."""
    table = Table(title=f"Validation — {vresult.summary}")
    table.add_column("Check", style="cyan")
    table.add_column("Result", style="bold")
    table.add_column("Message")
    for check in vresult.checks:
        icon = "[green]PASS[/green]" if check.passed else "[red]FAIL[/red]"
        table.add_row(check.name, icon, check.message)
    console.print(table)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments and run the harness."""
    parser = argparse.ArgumentParser(
        description="Kairos Prompt Development Harness — iterate on simulation prompts",
    )
    parser.add_argument(
        "--category",
        type=str,
        help="Scenario category to test",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the random seed",
    )
    parser.add_argument(
        "--concept",
        type=str,
        default=None,
        help="Path to a ConceptBrief JSON file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate and display the prompt without calling the LLM",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation after execution",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=1,
        help="Max adjustment iterations (default: 1 = no adjustment)",
    )
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="List available scenario categories and exit",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.list_categories:
        console.print("[bold]Available categories:[/bold]")
        for cat in ScenarioCategory:
            console.print(f"  • {cat.value}")
        return

    if not args.category:
        parser.error("--category is required (use --list-categories to see options)")

    concept_json = None
    if args.concept:
        concept_json = Path(args.concept).read_text(encoding="utf-8")

    asyncio.run(
        run_harness(
            args.category,
            seed=args.seed,
            concept_json=concept_json,
            dry_run=args.dry_run,
            skip_validation=args.skip_validation,
            max_iterations=args.max_iterations,
        )
    )


if __name__ == "__main__":
    main()
