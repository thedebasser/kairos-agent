"""Generate test config JSON files for manually exercising the Blender scripts.

Run from project root:
    python scripts/make_test_configs.py

Outputs to output/calibration/manual_test/ for inspection + manual Blender runs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "src"))

from kairos.calibration.models import (
    CorrectionFactors,
    PathDescriptor,
    PathType,
    ScenarioDescriptor,
)
from kairos.calibration.scenario import scenario_to_blender_config

OUT_DIR = _repo_root / "output" / "calibration" / "manual_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCENARIOS = [
    (
        "straight_30",
        ScenarioDescriptor(
            path=PathDescriptor(type=PathType.STRAIGHT, amplitude=0.0, cycles=1.0),
            domino_count=30,
        ),
    ),
    (
        "s_curve_gentle",
        ScenarioDescriptor(
            path=PathDescriptor(type=PathType.S_CURVE, amplitude=1.0, cycles=2.0),
            domino_count=300,
        ),
    ),
    (
        "spiral_2",
        ScenarioDescriptor(
            path=PathDescriptor(type=PathType.SPIRAL, spiral_turns=2.0),
            domino_count=200,
        ),
    ),
]


def main() -> None:
    print(f"\nWriting test configs to: {OUT_DIR}\n")
    for name, scenario in SCENARIOS:
        config = scenario_to_blender_config(scenario, CorrectionFactors())
        # Add a fixed seed for reproducible testing
        config["seed"] = 42

        out_path = OUT_DIR / f"{name}_config.json"
        out_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

        print(f"  {name}_config.json")
        print(f"    archetype        : {config['archetype']}")
        print(f"    domino_count     : {config['domino_count']}")
        print(f"    path_amplitude   : {config.get('path_amplitude', 'N/A')}")
        print(f"    spacing_ratio    : {config['spacing_ratio']}  (baseline=0.35)")
        print(f"    substeps_per_frame: {config['substeps_per_frame']}")
        print()

    print("=" * 60)
    print("MANUAL DRY RUN COMMANDS")
    print("(run from project root, with Blender on PATH)")
    print("=" * 60)
    for name, _ in SCENARIOS:
        cfg = OUT_DIR / f"{name}_config.json"
        blend = OUT_DIR / f"{name}.blend"
        gen_script = _repo_root / "src/kairos/engines/blender/scripts/generate_domino_course.py"
        smoke_script = _repo_root / "src/kairos/engines/blender/scripts/smoke_test_domino.py"
        print(f"\n# --- {name} ---")
        print(f"# Step 1: Generate")
        print(
            f"blender --background --python \"{gen_script}\" "
            f"-- --config \"{cfg}\" --output-blend \"{blend}\""
        )
        print(f"# Step 2: Smoke test (calibration mode)")
        print(
            f"blender --background \"{blend}\" --python \"{smoke_script}\" "
            f"-- --calibration-mode"
        )

    print("\n# --- Full calibration CLI dry run (all through Python) ---")
    print("python scripts/run_calibration.py --scenario straight_30 --dry-run --max-iterations 2")
    print("python scripts/run_calibration.py --scenario s_curve_gentle --dry-run --max-iterations 2")
    print()


if __name__ == "__main__":
    main()
