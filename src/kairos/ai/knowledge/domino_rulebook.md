# Domino Pipeline — Rulebook

Hard-won lessons from production runs. This file is injected into the LLM
concept-generation prompt so the model avoids repeating known failures.

---

## Path & Layout Rules

1. **No sharp turns (≥ 45°).** Domino chains cannot propagate around corners
   sharper than ~30°. The falling domino's momentum is directional — a 90°
   turn means the next domino is hit on its *side*, not its face, and the
   chain stalls. **Prefer smooth, sweeping curves at all times.**
   - BAD: zigzag rows with 90° switchbacks (cascade archetype row-ends)
   - BAD: right-angle "L" shapes, staircase patterns
   - GOOD: gentle S-curves (path_cycles ≤ 2.0), wide arcs, spirals

2. **Cascade archetype needs rounded row transitions.** If using "cascade"
   (wide zigzag rows filling the frame), the row-end transitions must be
   smooth U-turns with a minimum radius ≥ 5× domino spacing, NOT sharp
   corners. The path generator adds this automatically.

3. **Minimum curve radius: 5× spacing.** Any curve in the path must have a
   radius of curvature ≥ 5 × domino spacing (≈ 7 BU at default settings).
   Tighter curves cause dominos to miss each other or collide sideways.

## Physics Rules

4. **Do not change locked physics values.** Mass, friction, bounce,
   spacing_ratio, trigger impulse, and domino dimensions are research-locked.
   Changing them breaks the chain propagation.

5. **Trigger impulse must be gentle.** The default 1.5 is calibrated.
   Higher impulse launches the first domino instead of tipping it into
   the second.

## Visual / Camera Rules

6. **Camera tracking is automatic.** The pipeline uses a physics-aware
   tracking camera that follows the actual wave front post-bake. No need
   to design for a specific camera angle — just make the path interesting.

7. **Colour palette variety.** Alternate colours every 10-20 dominos for
   visual interest. The palette choice affects domino material colours.

## U-Turn & Corner Rules

8. **Cascade U-turns must be true semicircles.** Never squish the arc
   horizontally (e.g., `* 0.6`). A squished elliptical U-turn has very
   tight curvature at the apex, causing dominoes to collide sideways and
   "explode" at the start of each row. Use `radius * sin(angle)` without
   any damping factor.

9. **Scale cascade dimensions with domino count.** Fixed-size cascades
   (`width=50 BU`, `height=60 BU`) are too small for 200+ dominos. The
   path auto-scaler (Step 1c) then scales everything uniformly, which
   compresses the U-turns. Instead, compute
   `scale = max(1.0, count / 100.0)` and multiply both width and height.

10. **More arc points for smoother U-turns.** Use ≥ 20 interpolation
    points per semicircular arc (not 12). Fewer points create coarse
    polygon corners that the Chaikin smoother must fix — and may not fully
    resolve before resampling.

11. **Curvature-adaptive spacing on curves.** Uniform arc-length spacing
    is NOT enough — on a curve, adjacent dominos face each other at an
    angle, reducing the effective face-to-face gap. The correct formula:
    `spacing_curve = spacing_straight / cos(θ/2)` where θ is the local
    turning angle. This is implemented in `_resample_path`. Without this,
    dominos at U-turn apexes are laterally too close and "explode" from
    collision at physics start.

## SFX Rules

12. **Never FFmpeg-resample short synthetic sounds.** The
    `asetrate → aresample` pitch-shift pipeline introduces quantisation
    noise on sounds < 100 ms. Apply gain variation only (pure PCM
    scaling) and let the underlying synthesiser provide timbre variety
    through its own frequency jitter.

13. **Always include `[0:a]` in the final compositor.** The SFX audio
    is baked into `render.mp4` by `mix_audio.py`, but the FFmpeg
    compositor historically only mapped `[1:a]` (music) + TTS, silently
    dropping the collision sounds. The compositor *must* map `[0:a]`
    into the final `amix` — upmix mono→stereo with
    `aformat=channel_layouts=stereo` before mixing.

---

*Add new rules here as failures are discovered. Each rule should describe
the problem, why it happens, and what to do instead.*
