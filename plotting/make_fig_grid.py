#!/usr/bin/env python3
"""
make_figure_grid.py

Arrange a grid of figure files (.png, .pdf, .eps) into a single composite
PDF figure, with capital-letter panel labels (A, B, C, ...) placed just
above the top-left corner of each panel, assigned row-wise.

Panels supplied as .pdf or .eps are embedded into the output as native
vector content (not rasterized), so lines, curves, and text stay crisp
at any zoom level. .png panels are embedded as-is (they are raster to
begin with).

-----------------------------------------------------------------------
Requirements
-----------------------------------------------------------------------
    pip install pandas pymupdf pillow

Everything (vector PDF embedding, image placement, text labels) is done
with PyMuPDF alone -- no reportlab or pypdf needed.

.eps support additionally requires Ghostscript to be installed on your
system and available on PATH (used to convert EPS to PDF while keeping
it vector -- PyMuPDF itself cannot read EPS directly):
    - macOS:   brew install ghostscript
    - Ubuntu:  sudo apt-get install ghostscript
    - Windows: install from https://ghostscript.com/releases/gsdnld.html

Panel labels use Arial if it's installed on your system. Otherwise the
script automatically falls back to the metrically-identical Liberation
Sans, then DejaVu Sans, then PyMuPDF's built-in Helvetica.

-----------------------------------------------------------------------
Usage
-----------------------------------------------------------------------
Provide the grid as a string, rows separated by newlines (or the literal
two characters "\\n"), columns separated by commas:

    python make_figure_grid.py \\
        --grid "fig_1.eps,fig_2.eps
fig_3.eps,fig_4.eps" \\
        --output combined.pdf

or, on a single line (handy from a shell / config file):

    python make_figure_grid.py --grid "fig_1.eps,fig_2.eps\\nfig_3.eps,fig_4.eps" --output combined.pdf

This produces:

    A | B
    ---+---
    C | D

with A, B, C, D placed near the top-left of each panel, assigned
row-wise in the order the files were given. Labels are drawn outside the
panel bodies, in reserved space above each row.

If your figures all live in one folder, point --fig-dir at it instead of
repeating the path in every cell of --grid:

    python make_figure_grid.py \\
        --grid "fig_1.eps,fig_2.eps\\nfig_3.eps,fig_4.eps" \\
        --fig-dir /path/to/figures \\
        --output combined.pdf

Filenames in --grid that are already absolute paths are left as-is, so
you can mix figures from --fig-dir with figures elsewhere. Output
directories are created automatically if they don't already exist.

To leave a grid spot blank (e.g. 3 panels in a 2x2 layout), just leave
that cell empty between commas, or use "none"/"empty"/"-":

    python make_figure_grid.py \\
        --grid "fig_1.eps,fig_2.eps\\nfig_3.eps," \\
        --output combined.pdf

Blank spots get no panel and no letter -- letters are assigned only to
the actual panels, in row-wise order.
-----------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import io
import math
import os
import shutil
import string
import subprocess
import sys
import tempfile

import pandas as pd
from PIL import Image
import pymupdf as fitz


# --------------------------------------------------------------------------
# Grid string parsing
# --------------------------------------------------------------------------
# Cells matching any of these (case-insensitive, after stripping whitespace)
# are treated as intentionally empty grid spots: no panel is drawn there and
# no letter is assigned to it. A cell left blank between commas (e.g.
# "fig_1.png,,fig_3.png") works the same way.
EMPTY_MARKERS = {"", "none", "empty", "na", "n/a", "-", "nan"}


def is_empty_cell(cell):
    """True if `cell` denotes an intentionally empty grid spot."""
    return cell.strip().lower() in EMPTY_MARKERS


def parse_grid(grid_str):
    """
    Turn a grid string into a rectangular list-of-lists of filenames,
    using pandas to do the CSV-style parsing (rows on lines, columns
    comma-separated). Cells may be left empty (or set to "none"/"empty"/
    "-") to leave that grid spot blank -- pandas naturally represents
    those as NaN, which we convert to "".
    """
    # Accept either real newlines or a literal "\n" typed on one line.
    normalized = grid_str.replace("\\n", "\n")

    try:
        df = pd.read_csv(
            io.StringIO(normalized),
            header=None,
            dtype=str,
            skip_blank_lines=True,
            skipinitialspace=True,
        )
    except pd.errors.ParserError as e:
        raise ValueError(
            f"Could not parse --grid as a rectangular table: {e}. "
            f"Make sure every row has the same number of comma-separated "
            f"entries (use a blank cell, e.g. 'a.png,,c.png', for an "
            f"intentionally empty spot rather than dropping the comma)."
        )
    except pd.errors.EmptyDataError:
        raise ValueError("Grid string is empty.")

    df = df.fillna("").apply(lambda col: col.str.strip())

    if (df.map(is_empty_cell) if hasattr(df, "map") else df.applymap(is_empty_cell)).values.all():
        raise ValueError("Grid is entirely empty -- no panel files were given.")

    return df.values.tolist()


def resolve_grid_paths(grid, fig_dir=None):
    """
    Prefix each filename in the grid with `fig_dir`, unless the filename
    is already an absolute path or an intentionally empty cell. Returns a
    new grid of resolved paths.
    """
    if not fig_dir:
        return grid
    resolved = []
    for row in grid:
        resolved.append(
            [cell if (is_empty_cell(cell) or os.path.isabs(cell))
             else os.path.join(fig_dir, cell)
             for cell in row]
        )
    return resolved


# --------------------------------------------------------------------------
# Font handling (Arial, with graceful fallbacks)
# --------------------------------------------------------------------------
def find_label_font():
    """
    Look for a real Arial TTF on the system. If Arial isn't installed
    (it usually isn't on Linux), fall back to Liberation Sans or DejaVu
    Sans, both metrically compatible with / very close to Arial. Returns
    a path to a TTF file, or None to fall back to PyMuPDF's built-in
    Helvetica (also visually near-identical to Arial).
    """
    candidates = [
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


# --------------------------------------------------------------------------
# EPS -> vector PDF conversion (PyMuPDF can't read EPS directly)
# --------------------------------------------------------------------------
def eps_to_pdf(eps_path, tmp_dir):
    """Convert an EPS file to a PDF via Ghostscript, preserving vector data."""
    if shutil.which("gs") is None:
        raise RuntimeError(
            "Ghostscript ('gs') was not found on PATH. It's required to "
            "convert .eps panels to vector PDF. Install it (e.g. "
            "'apt-get install ghostscript' or 'brew install ghostscript') "
            "and try again."
        )
    out_path = os.path.join(
        tmp_dir, os.path.splitext(os.path.basename(eps_path))[0] + ".pdf"
    )
    cmd = [
        "gs", "-q", "-dBATCH", "-dNOPAUSE", "-dEPSCrop",
        "-sDEVICE=pdfwrite", f"-sOutputFile={out_path}", eps_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError(
            f"Ghostscript failed to convert '{eps_path}' to PDF:\n"
            f"{result.stderr}"
        )
    return out_path


# --------------------------------------------------------------------------
# Grid composition (vector-preserving, PyMuPDF only)
# --------------------------------------------------------------------------
@dataclass
class PanelSpec:
    """Metadata needed to place one panel without re-reading its dimensions."""
    path: str
    ext: str
    width_pt: float
    height_pt: float
    src_doc: fitz.Document | None = None


def load_panel_specs(grid, dpi, tmp_dir):
    """
    Resolve source dimensions in PDF points.

    PDF/EPS dimensions are already physical points. PNG dimensions are pixels,
    so `dpi` is used to infer their physical size. This keeps mixed PNG/PDF
    grids on a comparable scale when PNGs were exported at the requested DPI.
    """
    panel_specs = []
    open_src_docs = []

    for row in grid:
        spec_row = []
        for path in row:
            if is_empty_cell(path):
                spec_row.append(None)
                continue
            if not os.path.exists(path):
                raise FileNotFoundError(f"Figure file not found: {path}")

            ext = os.path.splitext(path)[1].lower()
            if ext == ".png":
                with Image.open(path) as img:
                    width_pt = img.width / dpi * 72.0
                    height_pt = img.height / dpi * 72.0
                spec_row.append(PanelSpec(path, ext, width_pt, height_pt))
            elif ext in (".pdf", ".eps"):
                src_path = eps_to_pdf(path, tmp_dir) if ext == ".eps" else path
                src_doc = fitz.open(src_path)
                open_src_docs.append(src_doc)
                src_page = src_doc[0]
                spec_row.append(
                    PanelSpec(
                        path=path,
                        ext=ext,
                        width_pt=src_page.rect.width,
                        height_pt=src_page.rect.height,
                        src_doc=src_doc,
                    )
                )
            else:
                raise ValueError(
                    f"Unsupported file type '{ext}' for '{path}'. "
                    f"Supported types: .png, .pdf, .eps"
                )
        panel_specs.append(spec_row)

    return panel_specs, open_src_docs


def compute_native_layout(panel_specs, max_width, max_height, wspace, hspace):
    """
    Compute a grid from native panel dimensions and fixed physical gutters.

    Column widths and row heights are chosen from the source panel aspect
    ratios. This makes the visible panels have equal width within each column
    and equal height within each row when panels are drawn with
    ``fit_mode='fill'``. For grids whose panel aspect ratios are not mutually
    compatible, this uses a small log-space least-squares solve so any required
    stretch is distributed rather than driven by an arbitrary nominal panel
    size.
    """
    nrows = len(panel_specs)
    ncols = len(panel_specs[0])
    gutter_x_pt = wspace * 72.0
    gutter_y_pt = hspace * 72.0

    base_col_widths = [
        max(
            (panel_specs[r][c].width_pt for r in range(nrows) if panel_specs[r][c]),
            default=0.0,
        )
        for c in range(ncols)
    ]
    base_row_heights = [
        max((spec.height_pt for spec in panel_specs[r] if spec), default=0.0)
        for r in range(nrows)
    ]

    panel_w_total = sum(base_col_widths)
    panel_h_total = sum(base_row_heights)
    if panel_w_total <= 0.0 or panel_h_total <= 0.0:
        raise ValueError("Grid is entirely empty -- no panel files were given.")

    occupied_by_col = [
        [r for r in range(nrows) if panel_specs[r][c]]
        for c in range(ncols)
    ]
    occupied_by_row = [
        [c for c in range(ncols) if panel_specs[r][c]]
        for r in range(nrows)
    ]

    # Solve log(col_width[c]) - log(row_height[r]) ~= log(src_w / src_h).
    # The additive degree of freedom is anchored to the native column widths.
    log_col_widths = [
        math.log(width) if width > 0.0 else 0.0
        for width in base_col_widths
    ]
    log_row_heights = [
        math.log(height) if height > 0.0 else 0.0
        for height in base_row_heights
    ]
    nonempty_cols = [c for c, rows in enumerate(occupied_by_col) if rows]
    anchor_log_col_width = (
        sum(log_col_widths[c] for c in nonempty_cols) / len(nonempty_cols)
        if nonempty_cols else 0.0
    )

    for _ in range(50):
        for c, rows in enumerate(occupied_by_col):
            if not rows:
                continue
            log_col_widths[c] = sum(
                math.log(panel_specs[r][c].width_pt / panel_specs[r][c].height_pt)
                + log_row_heights[r]
                for r in rows
            ) / len(rows)

        for r, cols in enumerate(occupied_by_row):
            if not cols:
                continue
            log_row_heights[r] = sum(
                log_col_widths[c]
                - math.log(panel_specs[r][c].width_pt / panel_specs[r][c].height_pt)
                for c in cols
            ) / len(cols)

        current_log_col_width = (
            sum(log_col_widths[c] for c in nonempty_cols) / len(nonempty_cols)
            if nonempty_cols else 0.0
        )
        offset = anchor_log_col_width - current_log_col_width
        for c in nonempty_cols:
            log_col_widths[c] += offset
        for r, cols in enumerate(occupied_by_row):
            if cols:
                log_row_heights[r] += offset

    col_widths = [
        math.exp(log_col_widths[c]) if occupied_by_col[c] else 0.0
        for c in range(ncols)
    ]
    row_heights = [
        math.exp(log_row_heights[r]) if occupied_by_row[r] else 0.0
        for r in range(nrows)
    ]

    max_w_pt = max_width * 72.0
    max_h_pt = max_height * 72.0
    fixed_gutters_w = (ncols - 1) * gutter_x_pt
    fixed_gutters_h = (nrows - 1) * gutter_y_pt
    available_w = max_w_pt - fixed_gutters_w
    available_h = max_h_pt - fixed_gutters_h
    if available_w <= 0.0 or available_h <= 0.0:
        raise ValueError(
            "Fixed panel spacing leaves no drawable area. Reduce --wspace/"
            "--hspace or increase --max-width/--max-height."
        )

    solved_w_total = sum(col_widths)
    solved_h_total = sum(row_heights)
    scale = min(available_w / solved_w_total, available_h / solved_h_total, 1.0)
    col_widths = [w * scale for w in col_widths]
    row_heights = [h * scale for h in row_heights]

    total_w_pt = sum(col_widths) + fixed_gutters_w
    total_h_pt = sum(row_heights) + fixed_gutters_h
    return col_widths, row_heights, gutter_x_pt, gutter_y_pt, total_w_pt, total_h_pt, scale


def compute_uniform_layout(
    nrows,
    ncols,
    panel_width,
    panel_height,
    max_width,
    max_height,
    wspace,
    hspace,
):
    """
    Compute the legacy fixed-cell grid, but with fixed physical gutters.
    """
    gutter_x_pt = wspace * 72.0
    gutter_y_pt = hspace * 72.0
    max_w_pt = max_width * 72.0
    max_h_pt = max_height * 72.0
    fixed_gutters_w = (ncols - 1) * gutter_x_pt
    fixed_gutters_h = (nrows - 1) * gutter_y_pt
    available_w = max_w_pt - fixed_gutters_w
    available_h = max_h_pt - fixed_gutters_h
    if available_w <= 0.0 or available_h <= 0.0:
        raise ValueError(
            "Fixed panel spacing leaves no drawable area. Reduce --wspace/"
            "--hspace or increase --max-width/--max-height."
        )

    fig_w_pt = ncols * panel_width * 72.0
    fig_h_pt = nrows * panel_height * 72.0
    scale = min(available_w / fig_w_pt, available_h / fig_h_pt, 1.0)
    panel_w_pt = panel_width * 72.0 * scale
    panel_h_pt = panel_height * 72.0 * scale
    col_widths = [panel_w_pt] * ncols
    row_heights = [panel_h_pt] * nrows
    total_w_pt = sum(col_widths) + fixed_gutters_w
    total_h_pt = sum(row_heights) + fixed_gutters_h
    return col_widths, row_heights, gutter_x_pt, gutter_y_pt, total_w_pt, total_h_pt, scale


def build_grid_pdf(
    grid,
    dpi=300,
    panel_width=3.0,
    panel_height=2.4,
    max_width=12.0,
    max_height=9.0,
    hspace=0.18,
    wspace=0.18,
    label_fontsize=12,
    start_letter="A",
    label_margin_pt=8.0,
    label_y_margin_pt=3.0,
    layout="native",
    fit_mode="fill",
    tmp_dir=None,
):
    """
    Build a composite PDF (as a pymupdf.Document) containing the panel
    grid, with vector-preserving placement of pdf/eps panels and letter
    labels drawn above the top-left of each placed panel.
    """
    nrows = len(grid)
    ncols = len(grid[0])

    panel_specs, open_src_docs = load_panel_specs(grid, dpi=dpi, tmp_dir=tmp_dir)

    # Reserve vertical space above every row so panel labels sit outside the
    # figure bodies instead of overlapping axes, titles, or plotted objects.
    # PyMuPDF positions text by baseline; 0.78 * fontsize is a close estimate
    # of the label's ascent and matches the previous top-margin calculation.
    label_ascent_pt = label_fontsize * 0.78
    label_band_pt = label_ascent_pt + max(label_y_margin_pt, 0.0)
    effective_max_height = max_height - (nrows * label_band_pt / 72.0)
    if effective_max_height <= 0.0:
        raise ValueError(
            "Panel labels leave no drawable height. Reduce --label-fontsize/"
            "--label-y-margin or increase --max-height."
        )

    if layout == "native":
        (
            col_widths,
            row_heights,
            gutter_x_pt,
            gutter_y_pt,
            total_w_pt,
            total_h_pt,
            scale,
        ) = compute_native_layout(
            panel_specs, max_width, effective_max_height, wspace, hspace
        )
    elif layout == "uniform":
        (
            col_widths,
            row_heights,
            gutter_x_pt,
            gutter_y_pt,
            total_w_pt,
            total_h_pt,
            scale,
        ) = compute_uniform_layout(
            nrows,
            ncols,
            panel_width,
            panel_height,
            max_width,
            effective_max_height,
            wspace,
            hspace,
        )
    else:
        raise ValueError(f"Unknown layout '{layout}'. Expected 'native' or 'uniform'.")

    total_h_pt += nrows * label_band_pt

    col_x0 = []
    x = 0.0
    for width in col_widths:
        col_x0.append(x)
        x += width + gutter_x_pt

    row_y0 = []
    y = 0.0
    for height in row_heights:
        y += label_band_pt
        row_y0.append(y)
        y += height + gutter_y_pt

    out_doc = fitz.open()
    page = out_doc.new_page(width=total_w_pt, height=total_h_pt)

    font_path = find_label_font()
    font_alias = "label_font"
    if font_path:
        page.insert_font(fontname=font_alias, fontfile=font_path)
    else:
        font_alias = "helv"  # PyMuPDF built-in, close to Arial

    letters = string.ascii_uppercase
    start_idx = letters.index(start_letter.upper())

    label_i = 0
    for r in range(nrows):
        for c in range(ncols):
            spec = panel_specs[r][c]
            if spec is None:
                # Intentionally blank grid spot: leave it blank, no letter.
                continue
            cell_llx = col_x0[c]
            cell_top_y = row_y0[r]  # PyMuPDF: y grows downward
            cell_rect_full = fitz.Rect(
                cell_llx,
                cell_top_y,
                cell_llx + col_widths[c],
                cell_top_y + row_heights[r],
            )

            if fit_mode == "fill":
                fit_rect = cell_rect_full
            elif fit_mode == "contain":
                fit_rect = _centered_fit_rect(cell_rect_full, spec.width_pt, spec.height_pt)
            else:
                raise ValueError(
                    f"Unknown fit mode '{fit_mode}'. Expected 'fill' or 'contain'."
                )

            if spec.ext == ".png":
                page.insert_image(
                    fit_rect,
                    filename=spec.path,
                    keep_proportion=(fit_mode == "contain"),
                )
            else:
                page.show_pdf_page(
                    fit_rect,
                    spec.src_doc,
                    0,
                    keep_proportion=(fit_mode == "contain"),
                )

            # Label near the top-left of the placed panel. This avoids labels
            # drifting away from narrow panels inside wider grid columns, while
            # keeping them outside the panel body.
            letter = letters[(start_idx + label_i) % len(letters)]
            label_x = fit_rect.x0 + label_margin_pt
            label_y = fit_rect.y0 - label_y_margin_pt
            page.insert_text(
                (label_x, label_y), letter,
                fontsize=label_fontsize, fontname=font_alias, color=(0, 0, 0),
            )

            label_i += 1

    return out_doc, open_src_docs


def _centered_fit_rect(cell_rect, src_w, src_h):
    """Return the largest rect centered in `cell_rect` matching src aspect."""
    cell_w, cell_h = cell_rect.width, cell_rect.height
    fit_scale = min(cell_w / src_w, cell_h / src_h)
    fit_w, fit_h = src_w * fit_scale, src_h * fit_scale
    x0 = cell_rect.x0 + (cell_w - fit_w) / 2.0
    y0 = cell_rect.y0 + (cell_h - fit_h) / 2.0
    return fitz.Rect(x0, y0, x0 + fit_w, y0 + fit_h)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Arrange figures into a labeled, vector-preserving grid PDF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--grid", required=True,
        help="Grid layout string. Rows separated by newline or literal "
             "'\\n', columns separated by commas, e.g. "
             "'fig_1.eps,fig_2.eps\\nfig_3.eps,fig_4.eps'",
    )
    parser.add_argument("--output", default="combined_figure.pdf",
                         help="Output PDF path (default: combined_figure.pdf). "
                              "Parent directories are created if needed.")
    parser.add_argument("--fig-dir", default=None,
                         help="Base directory where the panel files live. "
                              "Prepended to each filename in --grid unless "
                              "that filename is already an absolute path. "
                              "Default: current directory (paths in --grid "
                              "used as-is).")
    parser.add_argument("--dpi", type=int, default=300,
                         help="DPI used to infer physical size for PNG "
                              "panels (default: 300). PDF/EPS panels use "
                              "their native page dimensions.")
    parser.add_argument("--layout", choices=("native", "uniform"), default="native",
                         help="Panel layout mode. 'native' measures source "
                              "panels and derives row/column dimensions from "
                              "their aspect ratios; "
                              "'uniform' uses the legacy fixed-size cells "
                              "from --panel-width/--panel-height "
                              "(default: native).")
    parser.add_argument("--fit-mode", choices=("fill", "contain"), default="fill",
                         help="How panels are placed inside their row/column "
                              "cells. 'fill' makes every panel fill its cell, "
                              "so panels in the same row have equal visible "
                              "height and panels in the same column have equal "
                              "visible width; this can mildly stretch panels "
                              "when aspect ratios conflict. 'contain' preserves "
                              "source aspect ratios exactly but may leave "
                              "whitespace inside cells (default: fill).")
    parser.add_argument("--panel-width", type=float, default=3.0,
                         help="Nominal cell width in inches for "
                              "--layout uniform only (default: 3.0)")
    parser.add_argument("--panel-height", type=float, default=2.4,
                         help="Nominal cell height in inches for "
                              "--layout uniform only (default: 2.4)")
    parser.add_argument("--max-width", type=float, default=12.0,
                         help="Max overall figure width in inches, "
                              "used for page-fit scaling (default: 12.0)")
    parser.add_argument("--max-height", type=float, default=9.0,
                         help="Max overall figure height in inches "
                              "(default: 9.0)")
    parser.add_argument("--label-fontsize", type=float, default=12,
                         help="Font size for panel letters (default: 12)")
    parser.add_argument("--label-margin", type=float, default=8.0,
                         help="Horizontal distance in points from the panel's "
                              "left edge to the label (default: 8.0)")
    parser.add_argument("--label-y-margin", type=float, default=3.0,
                         help="Vertical gap in points between the label "
                              "baseline and the panel top edge "
                              "(default: 3.0; larger values move the letter "
                              "farther above the panel)")
    parser.add_argument("--start-letter", default="A",
                         help="Letter to start labeling from (default: A)")
    parser.add_argument("--hspace", type=float, default=0.18,
                         help="Fixed vertical spacing between panels in "
                              "inches (default: 0.18)")
    parser.add_argument("--wspace", type=float, default=0.18,
                         help="Fixed horizontal spacing between panels in "
                              "inches (default: 0.18)")

    args = parser.parse_args()

    try:
        grid = parse_grid(args.grid)
    except ValueError as e:
        print(f"Error parsing --grid: {e}", file=sys.stderr)
        sys.exit(1)

    grid = resolve_grid_paths(grid, args.fig_dir)

    print(f"Parsed a {len(grid)}x{len(grid[0])} grid:")
    for row in grid:
        display_row = [cell if not is_empty_cell(cell) else "(empty)" for cell in row]
        print("  ", display_row)

    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            out_doc, open_src_docs = build_grid_pdf(
                grid,
                dpi=args.dpi,
                panel_width=args.panel_width,
                panel_height=args.panel_height,
                max_width=args.max_width,
                max_height=args.max_height,
                hspace=args.hspace,
                wspace=args.wspace,
                label_fontsize=args.label_fontsize,
                label_margin_pt=args.label_margin,
                label_y_margin_pt=args.label_y_margin,
                start_letter=args.start_letter,
                layout=args.layout,
                fit_mode=args.fit_mode,
                tmp_dir=tmp_dir,
            )
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        out_dir = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(out_dir, exist_ok=True)

        out_doc.save(args.output, garbage=4, deflate=True)
        out_doc.close()
        for d in open_src_docs:
            d.close()

    print(f"Saved combined figure to: {args.output}")


if __name__ == "__main__":
    main()
