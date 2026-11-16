"""Layout preview of Fig. 1 at the final size (a placement reference for Illustrator).

    python -m paper_figures.fig01.preview

Places the exported components (their 600 dpi PNGs) at the positions recorded
in ``out/fig01/manifest.json`` on the exact 3.5 x 2.45 in canvas: panel (a)
on top, the four tiles below (for each slot the real tile, else the
provisional one, else the placeholder), the horizontal lambda colorbar and the
panel labels. Writes ``out/fig01/preview/fig01_layout_preview.{pdf,svg,png}``
(the 3D surface with the rod column) and two variants (``_ridgeline``: the
stacked curves instead of the surface; ``_obstacle``: the landscape with the
obstacle).
"""
from __future__ import annotations

import paper_figures  # noqa: F401

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
import numpy as np                                    # noqa: E402
import yaml                                           # noqa: E402
from PIL import Image                                 # noqa: E402

from paper_figures import style                       # noqa: E402
from paper_figures.common.export import Exporter      # noqa: E402
from paper_figures.common.paths import REPO_ROOT      # noqa: E402

CONFIG_PATH = Path(__file__).with_name("config.yaml")


def place(fig, entry, label=None):
    """Draw a component's PNG at its manifest placement (inches, lower-left)."""
    W, H = fig.get_size_inches()
    w, h = entry["canvas_in"]
    x, y = entry["placement"]["x_in"], entry["placement"]["y_in"]
    ax = fig.add_axes([x / W, y / H, w / W, h / H])
    ax.set_axis_off()
    with Image.open(REPO_ROOT / entry["files"]["png"]) as im:
        ax.imshow(np.asarray(im.convert("RGBA")), extent=(0, w, 0, h), interpolation="antialiased")
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_gid(label or entry["stem"])
    return ax


def pick_tile(comps, task):
    for stem in (f"b_tile_{task}_label", f"b_tile_{task}_provisional_label"):
        e = comps.get(stem)
        if e and e["status"] != "placeholder":
            return e
    return comps[f"b_tile_{task}_label"]


def compose(cfg, comps, panel_a, rods):
    style.apply()
    W, H = cfg["figure"]["size_in"]
    fig = plt.figure(figsize=(W, H))
    used = [panel_a, rods]
    place(fig, comps[panel_a])
    place(fig, comps[rods])
    cb = comps["colorbar_lambda_horizontal"]
    place(fig, cb)
    used.append(cb["stem"])
    for task in cfg["panel_b"]["tasks"]:
        e = pick_tile(comps, task)
        place(fig, e)
        used.append(e["stem"])
    lab = cfg["preview"]["labels_in"]
    for name, (x, y) in lab.items():
        t = fig.text(x / W, y / H, name, ha="left", va="top", fontsize=style.LABEL_SIZE,
                     fontweight="bold")
        t.set_gid(f"panel_label_{name.strip('()')}")
    return fig, used


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(CONFIG_PATH))
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    style.apply()
    ex = Exporter(cfg["figure"]["name"], cfg["figure"]["size_in"])
    ex.remove(f"{cfg['figure']['name']}_layout_preview")
    comps = ex.manifest["components"]
    variants = {"": ("a_surface_axes", "a_rods_column_label"),
                "_ridgeline": ("a_ridgeline_axes", "a_rods_column_label"),
                "_obstacle": ("a_surface_obstacle_axes", "a_rods_obstacle_column_label")}
    for suffix, (panel_a, rods) in variants.items():
        missing = [c for c in (panel_a, rods) if c not in comps]
        if missing:
            print(f"  skip preview{suffix}: {missing} not exported")
            continue
        fig, used = compose(cfg, comps, panel_a, rods)
        stem = f"{cfg['figure']['name']}_layout_preview{suffix}"
        ex.save(fig, stem, variant="preview", folder="preview",
                placement={"x_in": 0.0, "y_in": 0.0, "anchor": "lower-left"},
                status="preview", background="white",
                notes="placement reference only (raster components, on white)",
                extra={"components": used})
    print(json.dumps({s: ex.manifest["components"][s]["components"] for s in ex.saved}, indent=1))


if __name__ == "__main__":
    main()
