"""``save_component()``: PDF + SVG + PNG at the final size, a manifest entry, QA.

Every component goes through :meth:`Exporter.save`, which

1. refuses text below 7 pt (checked on the figure, before saving);
2. writes ``<stem>.pdf`` (vector, TrueType fonts), ``<stem>.svg`` (text kept as
   ``<text>``) and ``<stem>.png`` (600 dpi), all with a transparent background
   and the exact canvas (never ``bbox_inches='tight'``);
3. runs :mod:`checks` on the three files and raises on any violation;
4. records the component in ``out/<fig>/manifest.json``: canvas size,
   intended placement in the final figure, data scale and SVG group ids.

Variants: ``bare`` / ``graphic`` carry no text and no clipping masks;
``axes`` / ``label`` / ``preview`` carry text.
"""
from __future__ import annotations

import json

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.text import Text

from paper_figures import style
from paper_figures.common import checks
from paper_figures.common.paths import REPO_ROOT, fig_dirs, rel

TEXT_FREE = ("bare", "graphic")


def figure_gids(fig):
    return sorted({a.get_gid() for a in fig.findobj() if a.get_gid()})


def small_text(fig):
    return [(t.get_text(), t.get_fontsize()) for t in fig.findobj(Text)
            if t.get_text().strip() and t.get_fontsize() < style.MIN_FONT_SIZE - 1e-9]


def outside_canvas(fig, tol_pt=1.0):
    """Visible text and 2D data artists (with a gid) that extend past the canvas."""
    from mpl_toolkits.mplot3d import Axes3D
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    tol = tol_pt * fig.dpi / 72.0
    W, H = fig.bbox.width, fig.bbox.height
    out = []
    cands = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()
             and (t.axes is None or t.axes.axison or t in t.axes.texts)]
    cands += [a for a in fig.findobj() if a.get_gid() and not isinstance(a, Text)
              and getattr(a, "axes", None) is not None and not isinstance(a.axes, Axes3D)]
    for a in cands:
        try:
            bb = a.get_window_extent(r)
        except Exception:                                  # noqa: BLE001
            continue
        if not (bb.width or bb.height) or not np.all(np.isfinite(bb.extents)):
            continue
        if bb.x0 < -tol or bb.y0 < -tol or bb.x1 > W + tol or bb.y1 > H + tol:
            name = a.get_text() if isinstance(a, Text) else a.get_gid()
            out.append(f"{name!r} extends past the canvas "
                       f"({bb.x0 / fig.dpi:.3f}, {bb.y0 / fig.dpi:.3f}, {bb.x1 / fig.dpi:.3f}, "
                       f"{bb.y1 / fig.dpi:.3f} in)")
    return out


class Exporter:
    """Writes components of one figure and keeps its manifest."""

    def __init__(self, fig_name: str, figure_size_in):
        self.fig_name = fig_name
        self.dirs = fig_dirs(fig_name)
        self.path = self.dirs["base"] / "manifest.json"
        if self.path.exists():
            self.manifest = json.loads(self.path.read_text())
        else:
            self.manifest = {"figure": fig_name, "components": {}}
        self.manifest["figure"] = fig_name
        self.manifest["canvas_in"] = [float(v) for v in figure_size_in]
        self.manifest["placement_origin"] = (
            "x_in, y_in: lower-left corner of the component on the final figure "
            "canvas, inches from the figure's lower-left corner (y up)")
        self.saved = []

    def set(self, key, value):
        self.manifest[key] = value
        self.write()

    def write(self):
        comps = self.manifest.get("components", {})
        self.manifest["components"] = {k: comps[k] for k in sorted(comps)}
        self.path.write_text(json.dumps(self.manifest, indent=1, default=float) + "\n")

    def remove(self, prefix):
        """Drop the components whose stem starts with ``prefix``: manifest entries,
        their files and their annotations (a step clears its own outputs before
        exporting them again, so renamed components leave nothing stale)."""
        comps = self.manifest.get("components", {})
        stale = [k for k in comps if k.startswith(prefix)]
        for k in stale:
            for f in comps[k].get("files", {}).values():
                (REPO_ROOT / f).unlink(missing_ok=True)
            del comps[k]
        ann = self.dirs["data"] / "annotations.json"
        if stale and ann.exists():
            data = json.loads(ann.read_text())
            keep = {k: v for k, v in sorted(data.items()) if k not in stale}
            ann.write_text(json.dumps(keep, indent=1, default=float) + "\n")
        self.write()
        return stale

    def save(self, fig, stem, *, variant, placement=None, data_scale=None,
             status="final", notes=None, extra=None, folder="components", close=True,
             background=None):
        """Write ``stem`` as PDF + SVG + PNG and record it. ``background``: None
        (transparent, every component) or a color for layout previews, which
        are looked at in viewers that may ignore transparency."""
        size_in = [round(float(v), 6) for v in fig.get_size_inches()]
        small = small_text(fig)
        if small:
            raise checks.ComponentCheckError(f"{stem}: text below 7 pt: {small}")
        outside = outside_canvas(fig)
        if outside:
            raise checks.ComponentCheckError(f"{stem}: clipped by the canvas edge:\n  "
                                             + "\n  ".join(outside))
        out_dir = self.dirs[folder]
        files = {}
        for fmt in ("pdf", "svg", "png"):
            path = out_dir / f"{stem}.{fmt}"
            kw = dict(format=fmt, dpi=600)
            if background is None:
                kw["transparent"] = True
            else:
                kw["facecolor"] = background
            if fmt == "pdf":
                kw["metadata"] = {"CreationDate": None, "Creator": "paper_figures"}
            elif fmt == "svg":
                kw["metadata"] = {"Date": None, "Creator": "paper_figures"}
            fig.savefig(path, **kw)
            files[fmt] = path
        expect_text = variant not in TEXT_FREE
        violations, qa = checks.check_files(files, size_in, expect_text=expect_text,
                                            bare=variant in TEXT_FREE,
                                            opaque=background is not None)
        checks.assert_ok(stem, violations)
        entry = {
            "stem": stem,
            "files": {fmt: rel(p) for fmt, p in files.items()},
            "canvas_in": size_in,
            "variant": variant,
            "has_text": expect_text,
            "status": status,
            "placement": placement,
            "data_scale": data_scale,
            "gids": figure_gids(fig),
            "qa": qa,
            "background": background,
        }
        if notes:
            entry["notes"] = notes
        if extra:
            entry.update(extra)
        self.manifest.setdefault("components", {})[stem] = entry
        self.write()
        self.saved.append(stem)
        print(f"  saved {stem:44s} {size_in[0]:.3f} x {size_in[1]:.3f} in  "
              f"[{variant}{', ' + status if status != 'final' else ''}]  QA ok")
        if close:
            plt.close(fig)
        return entry

    # -------------------------------------------------------- annotations

    def annotate(self, stem, points, note=None):
        """Record annotation points of a component in ``data/annotations.json``.

        ``points``: ``{name: (x_in, y_in, extra_dict)}`` in the component's inch
        frame (origin lower-left, y up); ``y_in_from_top`` is added for
        Illustrator's top-left rulers.
        """
        path = self.dirs["data"] / "annotations.json"
        data = json.loads(path.read_text()) if path.exists() else {}
        h = self.manifest["components"].get(stem, {}).get("canvas_in", [None, None])[1]
        entry = {"frame": "inches from the component's lower-left corner, y up",
                 "points": {}}
        if note:
            entry["note"] = note
        for name, (x, y, extra) in points.items():
            rec = {"x_in": round(float(x), 4), "y_in": round(float(y), 4)}
            if h is not None:
                rec["y_in_from_top"] = round(float(h - y), 4)
            rec.update(extra or {})
            entry["points"][name] = rec
        data[stem] = entry
        path.write_text(json.dumps({k: data[k] for k in sorted(data)}, indent=1, default=float) + "\n")


def data_to_inches(fig, ax, x, y, z=None):
    """Data point -> inches from the figure's lower-left corner (draw the figure first)."""
    if z is not None:
        from mpl_toolkits.mplot3d import proj3d
        x, y, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
    disp = ax.transData.transform((x, y))
    return float(disp[0] / fig.dpi), float(disp[1] / fig.dpi)
