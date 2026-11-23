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
import io
import json
import re
import xml.etree.ElementTree as ET
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
    """The bare and label entries of the tile of ``task``: the real one, else the
    provisional one, else the placeholder."""
    for base in (f"b_tile_{task}", f"b_tile_{task}_provisional"):
        e = comps.get(f"{base}_bare")
        if e and e["status"] != "placeholder":
            return [e, comps[f"{base}_label"]]
    return [comps[f"b_tile_{task}_bare"], comps[f"b_tile_{task}_label"]]


def _prefixed_markup(inner, prefix):
    """``inner`` SVG markup with every id (and reference to it) prefixed, so several
    components can share one document."""
    inner = re.sub(r'\bid="([^"]+)"', lambda m: f'id="{prefix}__{m.group(1)}"', inner)
    return re.sub(r'(url\(#|href="#)([^")]+)', lambda m: f"{m.group(1)}{prefix}__{m.group(2)}", inner)


_SVG = "http://www.w3.org/2000/svg"
ET.register_namespace("", _SVG)
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
_NO_REMAINDER = ("_text", "_label", "colorbar_lambda_horizontal_axes", "colorbar_lambda_vertical_axes")


def _lift_texts(svg_text):
    """Split a component SVG into its ``<text>`` elements, each with the group that
    matplotlib wrapped it in (``[(id, xml)]``), and the markup that remains."""
    root = ET.fromstring(svg_text)
    parent = {c: p for p in root.iter() for c in p}
    lifted = []
    for t in [e for e in root.iter() if e.tag == f"{{{_SVG}}}text"]:
        g = parent[t]
        unit = g if g.tag == f"{{{_SVG}}}g" and g.get("id") and len(g) == 1 else t
        lifted.append((unit.get("id") or "".join(t.itertext()).strip() or "text",
                       ET.tostring(unit, encoding="unicode")))
        parent[unit].remove(unit)
    for md in [e for e in root.iter() if e.tag == f"{{{_SVG}}}metadata"]:
        parent[md].remove(md)
    rest = "".join(ET.tostring(c, encoding="unicode") for c in root)
    return lifted, rest


def write_layered_svg(path, cfg, comps, used):
    """The preview as an SVG with nothing fixed together: the graphics of every
    component are one group each (id = its stem), and every string of every
    component is a group of its own at the top level (id = its name, e.g.
    ``label_energy``), above the graphics. Illustrator opens top-level groups as
    layers, so each label can be selected, moved and edited on its own. Text stays
    ``<text>``; the panel labels are separate groups too."""
    W, H = cfg["figure"]["size_in"]
    graphics, texts, seen = [], [], {}
    for stem in used:
        e = comps[stem]
        w, h = e["canvas_in"]
        x, y = e["placement"]["x_in"], e["placement"]["y_in"]
        tr = f'translate({x * 72:g},{(H - y - h) * 72:g})'
        lifted, rest = _lift_texts((REPO_ROOT / e["files"]["svg"]).read_text())
        if not stem.endswith(_NO_REMAINDER):
            graphics.append(f'<g id="{stem}" transform="{tr}">{_prefixed_markup(rest, stem)}</g>')
        for name, xml in lifted:
            n = seen[name] = seen.get(name, 0) + 1
            gid = name if n == 1 else f"{name}__{stem}"
            texts.append(f'<g id="{gid}" transform="{tr}">{xml}</g>')
    for name, (x, y) in cfg["preview"]["labels_in"].items():
        texts.append(f'<g id="panel_label_{name.strip("()")}"><text x="{x * 72:g}" y="{y * 72:g}" '
                     f'font-family="{", ".join(style.FONT_SERIF)}, serif" '
                     f'font-size="{style.LABEL_SIZE:g}" font-weight="bold" '
                     f'dominant-baseline="hanging" fill="{style.INK}">{name}</text></g>')
    head = (f'<?xml version="1.0" encoding="utf-8"?>\n<svg xmlns="{_SVG}" '
            f'xmlns:xlink="http://www.w3.org/1999/xlink" width="{W * 72:g}pt" '
            f'height="{H * 72:g}pt" viewBox="0 0 {W * 72:g} {H * 72:g}" version="1.1">')
    bg = f'<rect id="background" width="{W * 72:g}" height="{H * 72:g}" fill="white"/>'
    Path(path).write_text("\n".join([head, bg, *graphics, *texts, "</svg>\n"]))


# one text as matplotlib writes it: q, placement, BT ... ET, Q
_BT_BLOCK = re.compile(rb"(?ms)^q\n[-\d.e ]+(?:\n[-\d.e ]+)? cm\nBT\n.*?^ET\nQ\n")


def write_merged_pdf(path, cfg, comps, used):
    """The preview as one PDF page built from the component PDFs themselves (no
    rasterizing), with **layers**: every string is wrapped in its own optional
    content group (Illustrator opens those as layers, named ``label_energy``,
    ``label_snap`` ...), every graphics component is one layer named after its stem,
    and the panel labels are layers too. Text stays live TrueType text. Needs
    ``pypdf``; returns False without."""
    try:
        from pypdf import PdfReader, PdfWriter, Transformation
        from pypdf.generic import (ArrayObject, DecodedStreamObject, DictionaryObject, NameObject,
                                   TextStringObject)
    except ImportError:
        print("  pypdf is not installed (pip install pypdf): the preview pdf stays flat")
        return False
    W, H = cfg["figure"]["size_in"]
    out = PdfWriter()
    page = out.add_blank_page(width=W * 72, height=H * 72)
    props = DictionaryObject()
    text_layers, graphic_layers, names_seen = [], [], {}

    def layer(name, bucket):
        n = names_seen[name] = names_seen.get(name, 0) + 1
        ocg = out._add_object(DictionaryObject({
            NameObject("/Type"): NameObject("/OCG"),
            NameObject("/Name"): TextStringObject(name if n == 1 else f"{name}__{n}")}))
        key = f"OC{len(props)}"
        props[NameObject("/" + key)] = ocg
        bucket.append(ocg)
        return f"/OC /{key} BDC\n".encode()

    def add(src, tr, stem, names, whole):
        data = src.get_contents().get_data()
        blocks = _BT_BLOCK.findall(data)
        if len(blocks) != len(names):
            names = [f"{stem}_text_{i + 1}" for i in range(len(blocks))]
        it = iter(names)
        data = _BT_BLOCK.sub(lambda m: layer(next(it), text_layers) + m.group(0) + b"EMC\n", data)
        if whole:
            data = layer(stem, graphic_layers) + data + b"EMC\n"
        st = DecodedStreamObject()
        st.set_data(data)
        src[NameObject("/Contents")] = st
        page.merge_transformed_page(src, tr)

    for stem in used:
        e = comps[stem]
        x, y = e["placement"]["x_in"], e["placement"]["y_in"]
        names = [n for n, _ in _lift_texts((REPO_ROOT / e["files"]["svg"]).read_text())[0]]
        src = PdfReader(str(REPO_ROOT / e["files"]["pdf"])).pages[0]
        add(src, Transformation().translate(x * 72, y * 72), stem, names,
            whole=not stem.endswith(_NO_REMAINDER))
    fig = plt.figure(figsize=(W, H))
    labs = cfg["preview"]["labels_in"]
    for name, (x, y) in labs.items():
        fig.text(x / W, y / H, name, ha="left", va="top", fontsize=style.LABEL_SIZE,
                 fontweight="bold")
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf", transparent=True)
    plt.close(fig)
    add(PdfReader(buf).pages[0], Transformation(), "panel_labels",
        [f"panel_label_{n.strip('()')}" for n in labs], whole=False)
    res = page["/Resources"].get_object() if "/Resources" in page else DictionaryObject()
    page[NameObject("/Resources")] = res
    have = res.get("/Properties")
    have = have.get_object() if have is not None else DictionaryObject()
    for k, v in props.items():
        have.setdefault(k, v)
    res[NameObject("/Properties")] = have
    top_down = text_layers + graphic_layers[::-1]       # text above the graphics
    out._root_object[NameObject("/OCProperties")] = DictionaryObject({
        NameObject("/OCGs"): ArrayObject(text_layers + graphic_layers),
        NameObject("/D"): DictionaryObject({NameObject("/Order"): ArrayObject(top_down),
                                            NameObject("/ON"): ArrayObject(top_down)})})
    with open(path, "wb") as f:
        out.write(f)
    return True


def compose(cfg, comps, layers):
    """``layers``: the stems of panel (a), stacked in order (graphics first, text last)."""
    style.apply()
    W, H = cfg["figure"]["size_in"]
    fig = plt.figure(figsize=(W, H))
    used = list(layers)
    for stem in layers:
        place(fig, comps[stem])
    cb = comps["colorbar_lambda_horizontal"]
    place(fig, cb)
    used.append(cb["stem"])
    for task in cfg["panel_b"]["tasks"]:
        for e in pick_tile(comps, task):
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
    variants = {"": ("a_surface_bare", "a_surface_axes", "a_surface_text",
                     "a_rods_column_bare", "a_rods_column_label"),
                "_ridgeline": ("a_ridgeline_bare", "a_ridgeline_axes",
                               "a_rods_column_bare", "a_rods_column_label"),
                "_obstacle": ("a_surface_obstacle_bare", "a_surface_obstacle_axes",
                              "a_surface_obstacle_text", "a_rods_obstacle_column_bare",
                              "a_rods_obstacle_column_label")}
    for suffix, layers in variants.items():
        missing = [c for c in layers if c not in comps]
        if missing:
            print(f"  skip preview{suffix}: {missing} not exported")
            continue
        fig, used = compose(cfg, comps, layers)
        stem = f"{cfg['figure']['name']}_layout_preview{suffix}"
        ex.save(fig, stem, variant="preview", folder="preview",
                placement={"x_in": 0.0, "y_in": 0.0, "anchor": "lower-left"},
                status="preview", background="white",
                notes="placement reference only (raster components, on white)",
                extra={"components": used})
        # svg and pdf are rewritten from the vector components (live text); only the png is flat
        files = ex.manifest["components"][stem]["files"]
        write_layered_svg(REPO_ROOT / files["svg"], cfg, comps, used)
        write_merged_pdf(REPO_ROOT / files["pdf"], cfg, comps, used)
    print(json.dumps({s: ex.manifest["components"][s]["components"] for s in ex.saved}, indent=1))


if __name__ == "__main__":
    main()
