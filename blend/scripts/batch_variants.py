"""Generate multiple variants from a base config.

Each variant tweaks cosmetic / small layout parameters while keeping
the archetype grammar fixed.  Seeds that fail validation or smoke test
are automatically retried with a new seed.

Run via:
    blender --background --python batch_variants.py -- --base-config <path.json> --count 20 --output-dir <dir>
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import bpy  # type: ignore[import-untyped]

_scripts_dir = str(Path(__file__).resolve().parent)
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from generate_course import generate_course  # noqa: E402
from validate_course import validate_course  # noqa: E402
from run_smoke_test import run_smoke_test  # noqa: E402
from bake_and_render import bake_and_render  # noqa: E402


# ---------------------------------------------------------------------------
# Variant parameter tweaking
# ---------------------------------------------------------------------------

def _make_variant_config(base_config: dict[str, Any], variant_seed: int,
                          tier: int = 1) -> dict[str, Any]:
    """Create a variant of *base_config* at the given tier.

    Tier 1: cosmetic only (colors, marble count small deltas)
    Tier 2: small layout changes (ramp slope, funnel width)
    Tier 3: different module sequence within grammar
    """
    rng = random.Random(variant_seed)
    cfg = {**base_config, "seed": variant_seed}

    # Load palettes
    configs_dir = Path(__file__).resolve().parent.parent / "configs"
    params_data = json.loads((configs_dir / "parameters.json").read_text())
    palettes = list(params_data["colour_palettes"].keys())
    lighting = list(params_data["lighting_presets"].keys())

    if tier >= 1:
        # Cosmetic: palette, count delta, lighting
        cfg["palette"] = rng.choice(palettes)
        cfg["lighting_preset"] = rng.choice(lighting)
        base_count = base_config.get("marble_count", 20)
        cfg["marble_count"] = base_count + rng.randint(-3, 5)

    if tier >= 2:
        # Small layout
        cfg["marble_radius"] = round(rng.uniform(0.03, 0.05), 4)
        cfg["marble_bounce"] = round(rng.uniform(0.3, 0.5), 2)
        # Don't set module_sequence — let it re-generate from archetype

    if tier >= 3:
        # New module sequence — clear any fixed sequence
        cfg.pop("module_sequence", None)

    return cfg


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_batch(base_config: dict[str, Any], count: int, output_dir: Path,
              *, tier: int = 1, max_retries: int = 3,
              render: bool = True, preset: str = "render_preview") -> dict[str, Any]:
    """Generate *count* variants, validate, smoke test, optionally render.

    Returns summary of pass/fail counts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    passed = 0
    failed = 0

    base_seed = base_config.get("seed", 42)

    for i in range(count):
        variant_seed = base_seed + i * 137  # deterministic spread
        attempt = 0
        success = False

        while attempt < max_retries and not success:
            seed = variant_seed + attempt * 7919
            variant_cfg = _make_variant_config(base_config, seed, tier=tier)
            attempt += 1

            print(f"\n{'='*60}")
            print(f"Variant {i+1}/{count} (seed={seed}, attempt={attempt})")
            print(f"{'='*60}")

            # Generate
            gen_result = generate_course(variant_cfg)
            print(f"  Generated: {gen_result['module_count']} modules, "
                  f"{gen_result['marble_count']} marbles")

            # Validate
            val_result = validate_course()
            print(f"  Validation: {val_result.summary}")
            if not val_result.passed:
                print(f"  FAILED validation — retrying")
                continue

            # Smoke test
            smoke_result = run_smoke_test(frames=120)
            print(f"  Smoke test: {'PASS' if smoke_result.passed else 'FAIL'}")
            if not smoke_result.passed:
                print(f"  FAILED smoke test: {smoke_result.reason} — retrying")
                continue

            success = True

            # Save blend
            blend_path = output_dir / f"variant_{i:03d}_seed{seed}.blend"
            bpy.ops.wm.save_as_mainfile(filepath=str(blend_path.resolve()))

            # Save config
            cfg_path = output_dir / f"variant_{i:03d}_seed{seed}.json"
            cfg_path.write_text(json.dumps({
                "config": variant_cfg,
                "generation": gen_result,
                "validation": val_result.to_dict(),
                "smoke_test": smoke_result.to_dict(),
            }, indent=2))

            # Render
            if render:
                video_path = output_dir / f"variant_{i:03d}_seed{seed}.mp4"
                render_result = bake_and_render(str(video_path), preset=preset)
                print(f"  Rendered: {render_result['output_path']} "
                      f"(bake={render_result['bake_time_sec']}s, "
                      f"render={render_result['render_time_sec']}s)")

            results.append({
                "variant": i,
                "seed": seed,
                "attempts": attempt,
                "passed": True,
                "modules": gen_result["module_count"],
                "marbles": gen_result["marble_count"],
            })

        if not success:
            failed += 1
            results.append({
                "variant": i,
                "seed": variant_seed,
                "attempts": attempt,
                "passed": False,
            })
        else:
            passed += 1

    summary = {
        "total": count,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / count * 100, 1) if count > 0 else 0,
        "variants": results,
    }

    # Save summary
    (output_dir / "batch_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True, help="Base config JSON path")
    parser.add_argument("--count", type=int, default=5, help="Number of variants")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--tier", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--preset", default="render_preview")
    args = parser.parse_args(argv)

    base_config = json.loads(Path(args.base_config).read_text())
    output_dir = Path(args.output_dir)

    summary = run_batch(
        base_config, args.count, output_dir,
        tier=args.tier,
        render=not args.no_render,
        preset=args.preset,
    )

    print(f"\n{'='*60}")
    print(f"Batch complete: {summary['passed']}/{summary['total']} passed "
          f"({summary['pass_rate']}%)")
    print(f"{'='*60}")
