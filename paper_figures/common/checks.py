"""QA of exported components: run after every export, and as a CLI.

    python -m paper_figures.common.checks              # every file in out/*/manifest.json

Per component (PDF, SVG, PNG):

* page / canvas size within 1 % of the intended size (no tight cropping);
* PDF fonts all embedded and none Type 3 (``pdffonts``, else a parse of the PDF);
* no text below 7 pt (SVG ``font`` sizes; the figure itself is checked before saving);
* SVG text stored as ``<text>`` elements (``svg.fonttype = 'none'``), and none in
  ``*_bare`` components;
* no clipping masks in ``*_bare`` components;
* transparent background (no opaque figure patch; PNG has alpha and clear pixels),
  except layout previews, which are drawn on white.

Violations are collected per file; :func:`assert_ok` raises with all of them.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

SIZE_TOL = 0.01                 # relative
MIN_PT = 7.0
PNG_DPI = 600
_SVG_NS = "{http://www.w3.org/2000/svg}"


class ComponentCheckError(RuntimeError):
    pass


# ------------------------------------------------------------------------ PDF

def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def pdf_page_size_pt(path):
    if shutil.which("pdfinfo"):
        m = re.search(r"Page size:\s+([\d.]+) x ([\d.]+) pts", _run(["pdfinfo", str(path)]))
        if m:
            return float(m.group(1)), float(m.group(2))
    raw = Path(path).read_bytes()
    m = re.search(rb"/MediaBox\s*\[\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\]", raw)
    x0, y0, x1, y1 = (float(v) for v in m.groups())
    return x1 - x0, y1 - y0


def pdf_fonts(path):
    """``[{name, type, emb}]``; from ``pdffonts`` if present, else parsed."""
    if shutil.which("pdffonts"):
        lines = _run(["pdffonts", str(path)]).splitlines()
        if len(lines) < 2:
            return []
        # column spans from the dashed separator line
        spans = [m.span() for m in re.finditer(r"-+", lines[1])]
        fonts = []
        for line in lines[2:]:
            cols = [line[a:b if i < len(spans) - 1 else None].strip()
                    for i, (a, b) in enumerate(spans)]
            fonts.append({"name": cols[0], "type": cols[1], "emb": cols[3] == "yes"})
        return fonts
    raw = Path(path).read_bytes()          # matplotlib writes font dicts uncompressed
    fonts = []
    for m in re.finditer(rb"/Subtype\s*/(Type3|TrueType|Type0|Type1)\b", raw):
        fonts.append({"name": "?", "type": m.group(1).decode(), "emb": True})
    n_desc = len(re.findall(rb"/Type\s*/FontDescriptor", raw))
    n_file = len(re.findall(rb"/FontFile[23]?\b", raw))
    if n_file < n_desc:
        fonts.append({"name": "?", "type": "?", "emb": False})
    return fonts


def check_pdf(path, size_in, expect_text):
    out = []
    w, h = pdf_page_size_pt(path)
    if not _close(w / 72.0, size_in[0]) or not _close(h / 72.0, size_in[1]):
        out.append(f"page {w / 72:.4f} x {h / 72:.4f} in, expected {size_in[0]} x {size_in[1]}")
    fonts = pdf_fonts(path)
    for f in fonts:
        if "Type 3" in f["type"] or f["type"] == "Type3":
            out.append(f"Type 3 font {f['name']}")
        if not f["emb"]:
            out.append(f"font not embedded: {f['name']}")
    if expect_text is False and fonts:
        out.append(f"bare component carries fonts: {[f['name'] for f in fonts]}")
    if expect_text and not fonts:
        out.append("no fonts although the component has text")
    return out, {"fonts": sorted({f"{f['name']} ({f['type']})" for f in fonts})}


# ------------------------------------------------------------------------ SVG

def _len_in(value):
    m = re.fullmatch(r"([\d.]+)(pt|in|px)?", value.strip())
    v, unit = float(m.group(1)), m.group(2) or "px"
    return v / 72.0 if unit in ("pt", "px") else v


def check_svg(path, size_in, expect_text, bare, opaque=False):
    out = []
    root = ET.parse(path).getroot()
    w, h = _len_in(root.get("width")), _len_in(root.get("height"))
    if not _close(w, size_in[0]) or not _close(h, size_in[1]):
        out.append(f"svg {w:.4f} x {h:.4f} in, expected {size_in[0]} x {size_in[1]}")
    texts = list(root.iter(f"{_SVG_NS}text"))

    def font_size(e):
        style = e.get("style", "") or ""
        m = (re.search(r"font-size:\s*([\d.]+)px", style)
             or re.search(r"font:\s*(?:[a-z-]+\s+)*([\d.]+)px", style))
        if m is None and e.get("font-size"):
            m = re.match(r"([\d.]+)", e.get("font-size"))
        return float(m.group(1)) if m else None

    sizes, unsized = [], 0
    for t in texts:
        # mathtext keeps its glyphs in <tspan> children that carry the size
        own = [font_size(e) for e in [t] + list(t.iter(f"{_SVG_NS}tspan"))]
        own = [v for v in own if v is not None]
        sizes += own
        unsized += not own
    small = [s for s in sizes if s < MIN_PT - 1e-6]
    if small:
        out.append(f"{len(small)} text element(s) below {MIN_PT} pt (min {min(small):.2f})")
    if expect_text and not texts:
        out.append("no <text> elements although the component has text")
    if expect_text is False and texts:
        out.append(f"bare component contains {len(texts)} <text> element(s)")
    if unsized:
        out.append(f"{unsized} text element(s) without a readable font size")
    glyph_defs = [e for e in root.iter(f"{_SVG_NS}path")
                  if re.match(r"(DejaVu|STIX|Times|Liberation|Nimbus)", e.get("id") or "")]
    if glyph_defs:
        out.append(f"{len(glyph_defs)} text glyph(s) converted to paths")
    n_clip = sum(1 for e in root.iter() if e.get("clip-path"))
    if bare and n_clip:
        out.append(f"bare component uses {n_clip} clipping mask(s)")
    for e in root.iter() if not opaque else ():
        if e.get("id") == "patch_1":           # the figure background patch
            for p in e.iter(f"{_SVG_NS}path"):
                st = p.get("style", "")
                if re.search(r"fill:\s*#[0-9a-fA-F]{6}", st) and "fill-opacity: 0" not in st:
                    out.append("opaque figure background in SVG")
    return out, {"n_text": len(texts), "font_sizes": sorted(set(sizes)), "n_clip": n_clip}


# ------------------------------------------------------------------------ PNG

def check_png(path, size_in, opaque=False):
    from PIL import Image
    out = []
    with Image.open(path) as im:
        wpx, hpx = im.size
        mode = im.mode
        alpha = im.getchannel("A") if "A" in mode else None
        hist = alpha.histogram() if alpha is not None else None
    want = (size_in[0] * PNG_DPI, size_in[1] * PNG_DPI)
    if not _close(wpx, want[0]) or not _close(hpx, want[1]):
        out.append(f"png {wpx} x {hpx} px, expected {want[0]:.0f} x {want[1]:.0f} at {PNG_DPI} dpi")
    clear = None
    if opaque:
        pass                                   # previews: white background by design
    elif hist is None:
        out.append(f"png has no alpha channel (mode {mode})")
    else:
        clear = hist[0] / float(wpx * hpx)
        if clear < 0.02:
            out.append(f"png background not transparent ({100 * clear:.1f}% clear pixels)")
    return out, {"px": [wpx, hpx], "clear_fraction": None if clear is None else round(clear, 3)}


# --------------------------------------------------------------------- driver

def _close(a, b, tol=SIZE_TOL):
    return abs(a - b) <= tol * max(abs(b), 1e-9)


def check_files(files, size_in, *, expect_text, bare, opaque=False):
    """Check ``{fmt: path}``; returns ``(violations, info)`` keyed by format.
    ``opaque``: a layout preview on a white background (no transparency check)."""
    violations, info = {}, {}
    for fmt, path in files.items():
        if fmt == "pdf":
            v, i = check_pdf(path, size_in, expect_text)
        elif fmt == "svg":
            v, i = check_svg(path, size_in, expect_text, bare, opaque)
        elif fmt == "png":
            v, i = check_png(path, size_in, opaque)
        else:
            continue
        if v:
            violations[fmt] = v
        info[fmt] = i
    return violations, info


def assert_ok(stem, violations):
    if violations:
        lines = [f"{stem}: QA failed"] + [f"  [{fmt}] {msg}" for fmt, msgs in violations.items()
                                          for msg in msgs]
        raise ComponentCheckError("\n".join(lines))


def check_manifest(manifest_path) -> int:
    """Re-check every component listed in a manifest; print a report."""
    from paper_figures.common.paths import REPO_ROOT
    manifest = json.loads(Path(manifest_path).read_text())
    n_bad = 0
    for stem, entry in sorted(manifest.get("components", {}).items()):
        files = {fmt: REPO_ROOT / p for fmt, p in entry["files"].items()}
        missing = [str(p) for p in files.values() if not p.exists()]
        if missing:
            print(f"FAIL {stem}: missing {missing}")
            n_bad += 1
            continue
        v, _ = check_files(files, entry["canvas_in"], expect_text=entry["has_text"],
                           bare=entry["variant"] == "bare",
                           opaque=entry.get("background") is not None)
        status = "ok  " if not v else "FAIL"
        n_bad += bool(v)
        print(f"{status} {stem:44s} {entry['canvas_in'][0]:.3f} x {entry['canvas_in'][1]:.3f} in")
        for fmt, msgs in v.items():
            for msg in msgs:
                print(f"       [{fmt}] {msg}")
    return n_bad


def main(argv=None):
    from paper_figures.common.paths import OUT_ROOT
    args = argv if argv is not None else sys.argv[1:]
    paths = [Path(a) for a in args] or sorted(OUT_ROOT.glob("*/manifest.json"))
    bad = 0
    for p in paths:
        p = p / "manifest.json" if p.is_dir() else p
        print(f"== {p}")
        bad += check_manifest(p)
    print(f"{bad} component(s) with violations")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
