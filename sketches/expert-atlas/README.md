# Expert Atlas Design Variants

Three throwaway HTML mockups exploring how users should interact with and view experts in JANG Studio's Expert Lab.

## Design goals

- **Simple and graphical**: experts as visual objects, not rows in a table.
- **Hover reveals semantic label**: domain, hits, drop risk, layer/expert ID.
- **Quick filter**: one-click filtering by domain (e.g. "show only safety experts") plus search and threshold.
- **Actionable**: select, mask, and retest without leaving the atlas.
- **Still exposes useful data**: detail panel keeps exact numbers, top prompts, comparison evidence.

## Variants

### 01 — Grid Atlas (`01-grid-atlas.html`)

A dense square grid: one cell per expert, rows are layers.

- **Strong at**: maximum information density, easy to see patterns across layers, compact.
- **Weak at**: small targets, harder to read expert IDs at a glance, can feel like a heatmap rather than objects.
- **Best for**: power users who want the full 256-expert landscape at once.

### 02 — Condensed Cards (`02-condensed-cards.html`)

Slightly larger cards per expert with ID, hit count, and a tiny activity bar.

- **Strong at**: more legible targets, each expert feels like a distinct object, hover/tap friendly.
- **Weak at**: less dense; a 256-expert model needs more scrolling or a wider viewport.
- **Best for**: most users — balances density with usability.

### 03 — Layer Strips (`03-layer-strip.html`)

Each layer is a single horizontal bar split into segments; hovering a segment expands it.

- **Strong at**: very compact vertically, emphasizes layer-level patterns, dramatic hover effect.
- **Weak at**: comparing experts across layers is harder, selecting multiple non-adjacent experts is awkward.
- **Best for**: overview-first workflows where layer behavior matters more than individual expert inspection.

## Common interactions across all variants

| Action | How it works |
|--------|--------------|
| Hover expert | Tooltip: domain, layer/expert ID, hits, tokens, router mass, drop risk |
| Click expert | Selects/deselects; detail panel updates |
| Double-click expert | Masks/unmasks that expert |
| Filter pills | Instantly filter to a domain or state (safety, hot, dead, masked, etc.) |
| Search | Filter by domain name, layer/expert ID, or hit count |
| Threshold slider | Hide low-activity experts |
| Color mode | Toggle domain vs. frequency vs. drop-risk coloring |
| Mask selected | Batch-mask all selected experts |
| Retest | Re-run the masked comparison |

## Recommendation

**Start with Variant 02 (Condensed Cards)**. It gives each expert enough visual presence to feel clickable, supports the requested hover+filter workflow cleanly, and scales reasonably for 256 experts per layer. Variant 01 can be kept as an optional "compact grid" toggle for power users. Variant 03 works best as a separate "layer overview" minimap rather than the primary interaction surface.

## Next steps

1. Pick a variant (or a hybrid).
2. Implement it in `ExpertLabSheet.swift`, replacing or supplementing the current grid/table toggle.
3. Wire the quick-filter pills to the existing `ExpertAtlasFilter` enum.
4. Make sure heavy filter/render work runs off the main thread (see Kanban task t_83b7fd36).
