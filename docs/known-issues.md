# Known Issues

## Phase 1 — Calibration System

### 3 Unresolved Bootstrap Scenarios

**Affected:** `spiral_4` (300 dominoes), `branching_3` (200 dominoes), `branching_5` (300 dominoes)

**Symptom:** All three scenarios fail with `SCRIPT_CRASH` — the Blender smoke test script (`smoke_test_domino.py`) crashes before producing any JSON output. Completion ratio is 0% on every iteration. The calibration sandbox now correctly detects this and breaks immediately instead of spinning for 10 iterations with empty corrections.

**Root Cause:** The `generate_domino_course.py` Blender script produces `.blend` files for these complex path types (branching layouts, 4-turn spirals) that cause the smoke test to crash. Likely causes:
- Missing or misconfigured rigid body world for branching/high-turn-count layouts
- Object naming assumptions in `smoke_test_domino.py` that don't hold for branching paths (multiple chains instead of one)
- Blender 5.x API differences in rigid body baking for complex scenes

**What's needed to fix:**
1. Run the failing `.blend` files in Blender GUI to identify the actual crash traceback
2. Update `generate_domino_course.py` to ensure rigid body world is correctly configured for branching and spiral layouts
3. Update `smoke_test_domino.py` to handle multi-chain layouts (branching creates separate domino chains per branch)
4. Re-run the 3 scenarios after fixes

### `cascade_400` Requires Human Review

**Affected:** `cascade_400` (400 dominoes)

**Symptom:** Physics passes on first iteration (smoke test + VLM both pass), but the quality gate flags it for human review instead of auto-promoting.

**Root Cause:** Expected behaviour. The quality gate requires `archetype_review_count >= AUTO_APPROVE_AFTER (20)` before auto-promoting. Since this is the first cascade archetype calibration, it's correctly gated. Not a bug.

**What's needed:** Manually approve via the review dashboard (`/web/review_app.py`) or lower `AUTO_APPROVE_AFTER` if the review confirms quality is acceptable.

### Perceptual Validator Renders Rest-Pose Frames

**Affected:** Any scenario where the VLM perceptual validator is invoked after a passing smoke test.

**Symptom:** The VLM reports "all dominoes remain upright" even when the smoke test confirms 100% chain completion. This triggers false perceptual failures. The 2-consecutive-failure safety valve eventually trusts the smoke test, but it wastes iterations.

**Root Cause:** `render_calibration_frames.py` opens the `.blend` and renders specific frames, but the rigid body simulation is only evaluated transiently via `scene.frame_set()` during the smoke test — the baked transforms aren't persisted into the `.blend` file. The render script sees the initial rest pose.

**What's needed:**
1. Update `render_calibration_frames.py` to either:
   - Re-bake the rigid body simulation before rendering, or
   - Use the same `scene.frame_set()` approach to step through physics before capturing each frame
2. Alternatively, bake the simulation into the `.blend` during the smoke test so subsequent scripts see the correct state
