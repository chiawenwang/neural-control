# paper_figures

Components of the TRO paper figures, built from the simulator in `src/` and
finished by hand in Adobe Illustrator. Every component is exported at the size
it has in the final figure (never scale it in Illustrator: that changes stroke
widths and font sizes), as PDF (vector, TrueType fonts), SVG (text kept as
`<text>`, artist groups named by `gid`) and PNG (600 dpi preview), with a
transparent background.

Nothing in `src/`, `scripts/` or `configs/` is modified; the package imports
`src/` the way the scripts do.

## Commands

Run from the repository root (`neural-control/`):

```bash
python -m paper_figures.fig01.landscape [--force] [--workers N]  # panel (a)
python -m paper_figures.fig01.tiles [--provisional] [--force]    # panel (b)
python -m paper_figures.fig01.preview                            # layout preview
python -m paper_figures.common.checks                            # re-run QA on everything
```

* `landscape` computes the physics once (about 13 min on 12 processes) and
  caches it in `cache/` under a hash of the physics settings in
  `fig01/config.yaml`; later runs only redraw. `--force` recomputes.
* `tiles` renders `data/rollouts/<task>/` for each task of `panel_b.tasks` and
  exports a placeholder where a task has no rollout yet. `--provisional` also
  builds the stand-ins described below (cached by their settings; `--force`
  rebuilds them) and renders them as `*_provisional`.
* `preview` assembles the exported components at their manifest placements,
  on a white background (some viewers ignore transparency and would show the
  translucent frames as opaque); the components themselves stay transparent.
* QA (`common/checks.py`) runs automatically after every export and raises on
  any violation.

All numeric settings are in `fig01/config.yaml`. There is no randomness: every
solve is deterministic, including the stability kicks.

## Inputs

* `src/` (simulator), `configs/default.yaml` (material presets),
  `configs/task_90deg_pipe.yaml` and `configs/example_3d_twist.yaml`
  (provisional tiles).
* `data/rollouts/<task>/` (tiles), in the standard rollout format below.

## Outputs

```
out/fig01/
  components/   <stem>.pdf / .svg / .png for every component
  preview/      fig01_layout_preview[_ridgeline|_obstacle].{pdf,svg,png}
  data/         curves, paths and surfaces as CSV / NPZ,
                landscape_summary.json (theta_c, snap, validation),
                annotations.json (points to annotate, in each component's inch frame)
  manifest.json one entry per component
cache/          cached computations (fig01_landscape_<hash>.pkl)
data/rollouts/  tile inputs (provisional stand-ins are written here too)
```

`manifest.json` records, per component: the files, the canvas in inches, the
variant (`bare`, `axes`, `label`, `graphic`, `preview`), the status (`final`,
`provisional`, `placeholder`), the intended placement on the 3.5 x 2.45 in
canvas (lower-left corner, inches from the figure's lower-left corner;
`alternative: true` marks a variant not used in the main layout), the data
scale (inches per unit, inches per meter for rods and tiles) and the SVG group
ids. `annotations.json` gives, for the same components, the coordinates of the
balls, the fold, the start and end of the snap jump and the gripper positions,
both from the bottom (`y_in`) and from the top (`y_in_from_top`, Illustrator's
default ruler).

## Fig. 1 components

Panel (a): the landscape at (0, 1.0), 3.5 x 1.45 in, its data area on the
left ~2.2 in; the rod column at (2.20, 1.0) on the right. Main layout: the
surface with the rod column (`fig01_layout_preview`).

| stem | canvas [in] | content |
|---|---|---|
| `a_surface_{bare,axes}` | 3.5 x 1.45 | `E(s, lambda)` surface (rasterized) with the realized path as vectors: balls joined by dashed links along the valley floor (the dashes are fitted to each gap), a thin arrow beside each branch (start to snap, landing to end), a curved arrow for the snap, and the saddle line (the barrier between the two valleys, ending at the fold where the snap starts) as dark dashes on a white halo. `axes` adds hand-placed axes (energy up the left edge, away from the rods) and the labels start, snap, end, saddle. Every arrow in panel (a), the rod column's too, has one line width (`style.LW_ARROW`) and head (`head_in`) |
| `a_rods_column_{bare,label}` | 1.3 x 1.45 | the process as a stack of rods (a flipbook), bottom to top, at the lambdas of the surface balls: start (`lambda = 0`), three frames on the way, snap (the last state before the jump), the first state after it, end (`lambda = 1`). Start, snap and end are solid and joined by arrows; the frames between are thin and translucent. Frames are spaced to fill the column and never touch, with a wider gap where the shape flips; `label` adds the names. `rods.column` in the config sets the opacity, spacing and direction (`down`: start at the top) |
| `a_rod_lam_<value>` | 1.1 x 0.5 | start, snap and end one per file (same scale as the column), clamps aligned with the end tangents |
| `a_rods` | 3.5 x 0.5 | the same three rods as one strip (alternative arrangement) |
| `a_ridgeline_{bare,axes}` | 3.5 x 1.45 | alternative style: `E(s; lambda_i)` for six lambdas, stacked by `lambda`; balls at the realized equilibria; valley-floor path colored by `lambda` (split at the snap); dashed gray line through the barrier maxima, ending at the fold |
| `a_ridgeline_path_solid` | 3.5 x 1.45 | the valley-floor path of the ridgeline as two single-color polylines (before / after the snap), for a gradient stroke |
| `colorbar_lambda_horizontal[_bare]` | 1.62 x 0.16 | `0 [bar] 1 lambda`, bar 1.2 x 0.08 in, between the panels |
| `colorbar_lambda_vertical[_bare]` | 0.40 x 1.30 | bar 0.08 x 1.0 in |
| `*_obstacle` | as above | the same with a circle above the arch |

The links, arrows and labels of the process are separate SVG groups (`link_k*`, `arrow_pre`,
`arrow_post`, `snap_arrow`, `arrow_start_snap`, `arrow_snap_end`, `label_start`,
`label_snap`, `label_end`, `label_saddle`; the saddle line is `saddle_line` over
`saddle_halo`), so they can be restyled or moved in Illustrator.

All rod sketches share one scale (`panel_a.rods.scale_in_per_m`, recorded in
the manifest). `s` is the midpoint height in units of `L`, energies are in
`EI/L`; the ridgeline offsets are arbitrary (no y tick labels), the offset per
unit `lambda` is in the manifest.

Panel (b): four tiles of 0.82 x 0.82 in in one row at y = 0
(`b_tile_<task>_{bare,label}`; `label` adds the task name).

| task | today |
|---|---|
| `shape_formation` | provisional: 2D, two `CircleObstacle` pegs, scripted gripper path |
| `insertion` | provisional: `configs/task_90deg_pipe.yaml`, scripted push (tilt, then feed) replayed through the pipe with contact |
| `routing` | provisional: `configs/example_3d_twist.yaml`, scripted gripper motion with a full turn of yaw (no pegs yet) |
| `hardware` | placeholder |

The provisional tiles are scripted replays, not optimized trajectories:
`train_pipe.py` saves no trajectory and its last logged run (60 epochs) took
27 min without reaching the target; the last logged 200-epoch twist run of
`train_point_to_point.py` took 78 min.

## Panel (a): what is computed

`fig01/landscape_compute.py`, through the adapters in `common/sim_adapters.py`:

1. `SimulatorEnv_2D`, both ends clamped `(1 - eps) L` apart (`eps = 0.1`),
   gravity off: the upper and lower arches from `+/- A sin^2(pi x / l)` guesses.
2. The right clamp is rotated, `theta_r = lambda theta_max`. Both directions are
   swept from the upper arch; the one whose branch ends is refined by
   bisection to `theta_c`; `theta_max = 1.3 theta_c`.
3. The realized path: 200 warm-started steps; exactly one snap is asserted
   (jump in `s` far above its median, or a spike in Newton iterations).
4. `E(s; lambda)` for lambda from 0 to 1 (the curves of the ridgeline and the
   rows of the surface): the midpoint's y DOF leaves `free_index` and is set to `s`;
   the rest is solved by `solve_equilibrium`, by continuation outward from each
   well, merged by the lowest energy. Every local minimum is released and
   matched to a stable unconstrained equilibrium; every barrier maximum to a
   saddle.
5. With `obstacle.enabled`, the same with a `CircleObstacle` above the arch;
   its penalty energy (the one whose force `src/contact.py` applies) is checked
   against those forces by finite differences.

Every accepted state is checked for stability (positive definite Hessian over
the solved DOFs): Newton returns any root, saddles included. An unstable state
is kicked along its unstable mode and the energy is minimized from there; when
warm-started Newton fails (past a fold), the energy is minimized from the warm
start (Newton with a shifted Hessian and an Armijo line search on the energy),
and the result is confirmed by `solve_equilibrium`.

## Standard rollout format (tiles)

Experiment scripts write one folder per task, `data/rollouts/<task>/`:

```
rollout.npz
  lam           (M,)         homotopy progress, nondecreasing in [0, 1]
  nodes         (M, N, dim)  node positions [m], dim = 2 or 3
  gripper_pose  (M, 3)       2D: x, y, theta [m, m, rad]
                (M, 7)       3D: x, y, z [m], quaternion qw, qx, qy, qz
                             (gripper frame: x = rod tangent into the gripper,
                             y = jaw direction)
  contact_mask  (M, N-1)     optional, bool: edge k (nodes k, k+1) touches an obstacle
scene.json
  dim, units ("m"), title, source, provisional,
  obstacles:   [{type: circle | sphere | cylinder | capsule | plane | pipe, ...}]
  fixed_end:   {position, tangent} or null
  gripper_end: "last" | "first"
  camera:      3D {elev, azim, zoom}; 2D optional {xlim, ylim}
```

Obstacle parameters (meters): `circle` {center, radius}; `sphere` {center,
radius}; `cylinder` / `capsule` {p0, p1, radius}; `plane` {point, normal,
size}; `pipe` {the `LPipe2D` parameters of `configs/*.yaml: pipe`, wall}.
`common/rollout_io.py` writes, reads and validates the format.

## Style

`style.py` holds the shared settings: Times-like serif (Times New Roman,
Times, STIXGeneral) with STIX math, 7-8 pt text, data strokes 0.8-1.2 pt,
auxiliary 0.5 pt, `LAMBDA_CMAP` (viridis truncated to [0.05, 0.92]) and
`lambda_color(lam)` for homotopy progress everywhere, light gray obstacles
with a dark gray outline, dark gray clamps, Okabe-Ito vermillion for
contacting edges.

## QA (`common/checks.py`)

Per exported file: page size within 1 % of the intended canvas; all PDF fonts
embedded and none Type 3 (`pdffonts`, else a parse of the PDF); no text below
7 pt; SVG text stored as `<text>`; no text in `bare` / `graphic` components and
no clipping masks there; transparent background. Before saving, the exporter
also refuses text below 7 pt and any text or 2D data artist that extends past
the canvas.
