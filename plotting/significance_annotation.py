import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties


def _annotate_significance_panel(
    ax,
    panel_df,
    x,
    y,
    yerr,
    hue,
    hue_order,
    x_labels,
    test_pairs,
    symbols,
    dodge_width=0.8,
    alpha=0.05,
    vertical_pad_frac=0.02,
    encode_strength=True,
    annotation_mode="bracket",  # "bracket" or "second_model"
    bracket_tick_frac=0.015,
    bracket_line_frac=0.01,
    group_stack_frac=0.06,  # vertical stacking between multiple comparisons in same x group
    annotation_fontsize="x-small",
    annotation_fontweight="normal",
):
    """
    Annotate a single panel (single Axes) with significance markers.

    annotation_mode:
        - "bracket": draw a bracket between the two bars and place the symbol centered above it
        - "second_model": place the symbol above the second model's bar

    This version keeps the bar geometry fixed from hue_order and only uses panel_df
    to decide whether a given bar exists. Missing bars do not change positions.
    """

    if annotation_mode not in {"bracket", "second_model"}:
        raise ValueError("annotation_mode must be either 'bracket' or 'second_model'")

    # ---------- Statistical helper ----------
    def p_from_z(z):
        return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))

    def is_significant(m1, se1, m2, se2):
        denom = math.sqrt(se1**2 + se2**2)
        if denom == 0:
            return False, 1.0
        z = (m1 - m2) / denom
        p = p_from_z(z)
        return p < alpha, p

    def symbol_from_p(p, base_symbol):
        if not encode_strength:
            return base_symbol
        if p < 0.001:
            return base_symbol * 3
        if p < 0.01:
            return base_symbol * 2
        if p < 0.05:
            return base_symbol
        return base_symbol

    # ---------- X positions ----------
    x_to_index = {lab: i for i, lab in enumerate(x_labels)}

    # ---------- Fixed dodge geometry from full hue_order ----------
    if hue and hue_order:
        num_hues = len(hue_order)
        bar_width = dodge_width / num_hues
        dodge_distances = np.linspace(
            -dodge_width / 2 + bar_width / 2,
            dodge_width / 2 - bar_width / 2,
            num_hues,
        )
    else:
        num_hues = 1
        bar_width = dodge_width
        dodge_distances = np.array([0.0])

    # ---------- Build lookup: full geometry, missing bars just become None ----------
    lookup = {}

    for x_lab in x_labels:
        if x_lab not in x_to_index:
            continue

        base_x = x_to_index[x_lab]

        if hue and hue_order:
            for i_h, hue_val in enumerate(hue_order):
                row = panel_df[(panel_df[x] == x_lab) & (panel_df[hue] == hue_val)]

                if row.empty:
                    lookup[(x_lab, hue_val)] = None
                else:
                    lookup[(x_lab, hue_val)] = {
                        "x": float(base_x + dodge_distances[i_h]),
                        "mean": float(row[y].iloc[0]),
                        "stderr": float(row[yerr].iloc[0]),
                    }
        else:
            row = panel_df[panel_df[x] == x_lab]
            if row.empty:
                lookup[(x_lab, None)] = None
            else:
                lookup[(x_lab, None)] = {
                    "x": float(base_x),
                    "mean": float(row[y].iloc[0]),
                    "stderr": float(row[yerr].iloc[0]),
                }

    # ---------- Collect annotations grouped by x category ----------
    group_annotations = {x_lab: [] for x_lab in x_labels}

    for x_lab in x_labels:
        for (m1, m2), base_symbol in zip(test_pairs, symbols):
            if hue and hue_order:
                if m1 not in hue_order or m2 not in hue_order:
                    continue

                entry1 = lookup.get((x_lab, m1))
                entry2 = lookup.get((x_lab, m2))
            else:
                entry1 = lookup.get((x_lab, None))
                entry2 = lookup.get((x_lab, None))

            if not entry1 or not entry2:
                continue

            m1_mean, m1_se = entry1["mean"], entry1["stderr"]
            m2_mean, m2_se = entry2["mean"], entry2["stderr"]

            sig, p = is_significant(m1_mean, m1_se, m2_mean, m2_se)
            if not sig:
                continue

            symbol = symbol_from_p(p, base_symbol)

            if annotation_mode == "second_model":
                x_anchor = entry2["x"]
                base_y = entry2["mean"] + entry2["stderr"]
                group_annotations[x_lab].append(
                    ("second_model", x_anchor, base_y, symbol)
                )
            else:
                x_left = entry1["x"]
                x_right = entry2["x"]
                if x_left > x_right:
                    x_left, x_right = x_right, x_left

                x_mid = 0.5 * (x_left + x_right)
                base_y = max(m1_mean + m1_se, m2_mean + m2_se)
                group_annotations[x_lab].append(
                    ("bracket", x_left, x_right, x_mid, base_y, symbol)
                )

    if not any(group_annotations.values()):
        return

    # ---------- Draw annotations ----------
    for x_lab in x_labels:
        items = group_annotations.get(x_lab, [])
        if not items:
            continue

        # Sort by comparison height so high-bar comparisons are placed first.
        def item_base_height(item):
            return item[2] if item[0] == "second_model" else item[4]

        items = sorted(items, key=item_base_height, reverse=True)

        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min if y_max > y_min else 1.0

        vpad = vertical_pad_frac * y_range
        tick_h = bracket_tick_frac * y_range
        line_gap = bracket_line_frac * y_range

        # Floor for all annotations in this x group: stay above every bar/errorbar top
        # so significance markers never overlap bars due to comparison ordering.
        bar_tops = []
        if hue and hue_order:
            for hue_val in hue_order:
                entry = lookup.get((x_lab, hue_val))
                if entry:
                    bar_tops.append(entry["mean"] + entry["stderr"])
        else:
            entry = lookup.get((x_lab, None))
            if entry:
                bar_tops.append(entry["mean"] + entry["stderr"])
        group_bar_top = max(bar_tops) if bar_tops else y_min
        annotation_floor = group_bar_top + vpad

        font_points = FontProperties(size=annotation_fontsize).get_size_in_points()

        def points_to_data_height(points, y_ref):
            display_xy = ax.transData.transform((0.0, y_ref))
            dy_pixels = points * ax.figure.dpi / 72.0
            y_shifted = ax.transData.inverted().transform(
                (display_xy[0], display_xy[1] + dy_pixels)
            )[1]
            return abs(y_shifted - y_ref)

        def annotation_text_height(y_ref):
            return max(points_to_data_height(font_points * 1.2, y_ref), 0.01 * y_range)

        # Keep this as a lower bound to preserve existing behavior, but collision
        # handling below will push labels further when needed.
        push_step = max(
            group_stack_frac * y_range,
            points_to_data_height(font_points * 1.0, y_max),
            1e-12,
        )
        min_gap = max(points_to_data_height(font_points * 0.25, y_max), 0.002 * y_range)
        occupied_spans = []

        def collides(proposed_span):
            low, high = proposed_span
            for prev_low, prev_high in occupied_spans:
                if (low < prev_high + min_gap) and (high > prev_low - min_gap):
                    return True
            return False

        for item in items:
            if item[0] == "second_model":
                _, x_anchor, base_y, symbol = item
                y_text = max(base_y + vpad, annotation_floor)
                text_h = annotation_text_height(y_text)
                span = (y_text, y_text + text_h)
                while collides(span):
                    y_text += push_step
                    span = (y_text, y_text + text_h)

                ax.text(
                    x_anchor,
                    y_text,
                    symbol,
                    fontsize=annotation_fontsize,
                    ha="center",
                    va="bottom",
                    color="grey",
                    fontweight=annotation_fontweight,
                )

                occupied_spans.append(span)
                needed_ymax = span[1] + 0.2 * text_h
                if needed_ymax > y_max:
                    y_max = needed_ymax

            else:
                _, x_left, x_right, x_mid, base_y, symbol = item
                y_bracket = max(base_y + vpad, annotation_floor)
                line_gap_dynamic = max(line_gap, points_to_data_height(font_points * 0.3, y_bracket))
                text_h = annotation_text_height(y_bracket)
                y_text = y_bracket + tick_h + line_gap_dynamic
                span = (y_bracket, y_text + text_h)
                while collides(span):
                    y_bracket += push_step
                    y_text = y_bracket + tick_h + line_gap_dynamic
                    span = (y_bracket, y_text + text_h)

                ax.plot(
                    [x_left, x_left],
                    [y_bracket, y_bracket + tick_h],
                    color="grey",
                    lw=1,
                )
                ax.plot(
                    [x_right, x_right],
                    [y_bracket, y_bracket + tick_h],
                    color="grey",
                    lw=1,
                )
                ax.plot(
                    [x_left, x_right],
                    [y_bracket + tick_h, y_bracket + tick_h],
                    color="grey",
                    lw=1,
                )

                ax.text(
                    x_mid,
                    y_text,
                    symbol,
                    fontsize=annotation_fontsize,
                    ha="center",
                    va="bottom",
                    color="grey",
                    fontweight=annotation_fontweight,
                )

                occupied_spans.append(span)
                needed_ymax = span[1] + 0.2 * text_h
                if needed_ymax > y_max:
                    y_max = needed_ymax

        ax.set_ylim(y_min, y_max)


def add_significance_annotations(
    plot_obj,
    data,
    x,
    y,
    yerr,
    hue,
    hue_order,
    x_labels,
    test_pairs,
    symbols=None,
    dodge_width=0.8,
    alpha=0.05,
    annotation_mode="bracket",  # "bracket" or "second_model"
    annotation_fontsize="x-small",
    annotation_fontweight="normal",
):
    """
    Works for single Axes or seaborn FacetGrid.
    """

    if symbols is None:
        symbols = ["*"] * len(test_pairs)

    # ---------- Single axis ----------
    if isinstance(plot_obj, plt.Axes):
        _annotate_significance_panel(
            plot_obj,
            data,
            x,
            y,
            yerr,
            hue,
            hue_order,
            x_labels,
            test_pairs,
            symbols,
            dodge_width=dodge_width,
            alpha=alpha,
            annotation_mode=annotation_mode,
            annotation_fontsize=annotation_fontsize,
            annotation_fontweight=annotation_fontweight,
        )
        return

    # ---------- FacetGrid ----------
    if hasattr(plot_obj, "axes_dict"):
        row_var = getattr(plot_obj, "_row_var", None) or getattr(
            plot_obj, "row_var", None
        )
        col_var = getattr(plot_obj, "_col_var", None) or getattr(
            plot_obj, "col_var", None
        )

        for key, ax in plot_obj.axes_dict.items():
            panel_df = data.copy()

            if isinstance(key, tuple):
                if row_var and col_var:
                    row_val, col_val = key
                    panel_df = panel_df[
                        (panel_df[row_var] == row_val) & (panel_df[col_var] == col_val)
                    ]
                elif col_var:
                    panel_df = panel_df[panel_df[col_var] == key[0]]
                elif row_var:
                    panel_df = panel_df[panel_df[row_var] == key[0]]
            else:
                if col_var:
                    panel_df = panel_df[panel_df[col_var] == key]
                elif row_var:
                    panel_df = panel_df[panel_df[row_var] == key]

            if panel_df.empty:
                continue

            _annotate_significance_panel(
                ax,
                panel_df,
                x,
                y,
                yerr,
                hue,
                hue_order,
                x_labels,
                test_pairs,
                symbols,
                dodge_width=dodge_width,
                alpha=alpha,
                annotation_mode=annotation_mode,
                annotation_fontsize=annotation_fontsize,
                annotation_fontweight=annotation_fontweight,
            )
