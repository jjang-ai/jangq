# JANG Studio Design Direction

The canonical visual reference for upcoming JANG Studio UI work is:

`JANGStudio/demos/design-b-v3.html`

**Intent Prune / quality loop** (capability shape → prove holdouts → convert) uses the same design-b language, with flow and copy fixed in:

`JANGStudio/demos/design-b-intent-prune-quality.html`

Influenced by `design-b-v2.html` (dock, drawer, workflow pills, domain colors, compare tray). Expert Lab atlas stays atlas-centric (v3); Intent Prune stays **capability-first** with a quality scoreboard and baseline vs pruned compare tray — not a second full atlas.

This direction is a compact pro-noir technical console, not a landing page or broad dashboard. The UI should feel like a dense model-inspection instrument: fast to scan, low-friction for repeated runs, and visually centered on the expert atlas (Expert Lab) or the gate scoreboard (Intent Prune).

## Core Layout

- Use a narrow top bar with the current mode, model badge, compact workflow rail, and path.
- Prefer a four-column work surface for Expert Lab:
  - slim icon dock
  - compact drawer for trace/probe/history/mask controls
  - atlas as the dominant center surface
  - right rail for selected expert evidence and actions
- Keep compare output in a bottom tray that appears when there is comparison data.
- Put live prompt probing close to the atlas so users can type a prompt and immediately see experts light up.
- Keep advanced settings collapsed by default.

## Visual Tokens

- Canvas: `#07090e`
- Panel: `#0e1316`
- Raised panel: `#131a1e`
- Hairline border: `rgba(255,255,255,0.05)`
- Primary accent: `#33e0e8`
- Warm/review action: `#ebb85e`
- Good/keep: `#70dba0`
- Danger/drop: `#ff5964`
- Primary text: `#e8edf0`
- Dim text: `#6a787d`
- Faint text: `#3a4548`

Use small radii: 2px for atlas cells, 3-4px for pills/buttons/inputs, 5-6px for panels. Avoid large floating cards, gradients, decorative blobs, oversized hero typography, or one-note blue/purple surfaces.

## Typography

- Favor compact system type and mono labels.
- Section labels should be uppercase, 9px-ish, medium weight, slight positive letter spacing.
- Operational values, paths, run IDs, and model metadata should use monospace styling.
- Keep copy terse. The interface should guide by layout and state, not by paragraphs of explanation.

## Expert Atlas

- The atlas is the main event. It should occupy most of the center view.
- Cells should be small, stable, square-ish, and densely packed.
- Use semantic domain colors for expert labels, with intensity indicating activation strength.
- Selection should be obvious with a bright border, but not visually noisy.
- Provide a compact legend on demand instead of permanently consuming atlas space.

## Interaction Defaults

- No automatic prompt runs on entry.
- Trace/probe actions must be explicit.
- Live prompt probing should create a normal trace artifact and feed the same atlas/evidence/mask/compare pipeline as suite probing.
- Empty atlas space should clearly indicate that prompts must be run to generate the expert map.
- Pruning actions stay gated behind trace evidence and A/B comparison.

## Intent Prune quality loop (product path)

Make the dogfood-proven process self-explanatory in the UI (see `design-b-intent-prune-quality.html`):

1. **Evidence** — one primary action: real-domain BF16/vMLX trace + transitions (no-thinking). Do not require hand-picking `expert_transitions.jsonl` as the happy path. Marker Reviewed 50 is routing-only, labeled as such.
2. **Shape** — keep chips, optional **drop** chips (e.g. Chinese), safety stance (Keep / Balanced / CRACK+confirm), size budget → keep-K.
3. **Prune** — hard-prune + structural verify only. Copy must say structural OK ≠ quality OK.
4. **Quality** — holdout scoreboard (math 5/5, code, EN, ZH/CJK, CRACK comply + still-refuse). Compare tray: baseline vs pruned answers (design-b-v2 tray pattern).
5. **Convert** — unlocked only when capability gates pass. Fail → suggest iterate (raise budget / more CRACK-pack traces).

Workflow pills: `01 Evidence → 02 Shape → 03 Prune → 04 Quality → 05 Convert`. Dock mirrors stages; Atlas remains Advanced Expert Lab escape hatch.

## SwiftUI Implementation Notes

- Centralize the palette and spacing in shared visual primitives before broad redesign work.
- Prefer thin, full-height rails and compact sections over nested card stacks.
- Replace text-heavy buttons with icon + concise verb where possible.
- Keep controls visually aligned to the workflow stage they affect.
- Visual QA should include the blank atlas state, live prompt run, atlas-with-selection state, mask/compare state, and prune-plan gated state.
