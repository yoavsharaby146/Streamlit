"""Streamlit combo viewer: Plot Data sessions + Angular Polar sessions.

Loads the JSON session files written by 'Plot data.py' and
'angular polar.py' and reproduces their core plotting features.
Run:  streamlit run streamlit_app.py
"""
import json
import os
import re

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==================== shared helpers (same logic as the Tk apps) ====================
ANGLE_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:deg|°)", re.IGNORECASE)
ANY_NUM_RE = re.compile(r"(-?\d+(?:\.\d+)?)")


def detect_angle_from_name(name):
    stem = name.rsplit(".", 1)[0].rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
    m = ANGLE_RE.search(stem)
    if m:
        return float(m.group(1))
    m = ANY_NUM_RE.search(stem)
    return float(m.group(1)) if m else None


def detect_angle(path):
    p = path.replace("\\", "/")
    folder = p.rstrip("/").rsplit("/", 2)[-2] if p.count("/") >= 2 else \
        (p.rstrip("/").split("/")[0] if "/" in p else "")
    a = detect_angle_from_name(folder) if folder else None
    fname = p.rsplit("/", 1)[-1] if "/" in p else p
    return a if a is not None else detect_angle_from_name(fname)


def value_at_x(xs, ys, x_target, mode="Linear interpolation"):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[mask], ys[mask]
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    if mode == "Nearest row":
        return float(ys[np.argmin(np.abs(xs - x_target))])
    return float(np.interp(x_target, xs, ys))


def format_angle_deg(deg, mode="0 to 360"):
    deg = float(deg) % 360.0
    if mode == "-180 to 180":
        return ((deg + 180.0) % 360.0) - 180.0
    if mode == "-90 to 90":
        return ((deg + 90.0) % 180.0) - 90.0
    return deg


THETA_ZERO_MAP = {"E (0 deg, standard)": "E", "N (top)": "N", "W": "W", "S": "S"}


def angular_polar_view(session):
    files = session.get("files", [])
    if not files:
        st.warning("No files inside this session.")
        return
    s = session.get("settings", {})

    # editable angle table
    def fname(p):
        return p.replace("\\", "/").rsplit("/", 1)[-1]

    rows = pd.DataFrame({
        "include": [bool(f.get("keep", True)) for f in files],
        "angle": [float(f.get("angle") or detect_angle(f.get("path", "")) or 0)
                  for f in files],
        "file": [fname(f.get("path", "?")) for f in files],
    })
    st.subheader("Files & angles (edit angle / untick include)")
    edited = st.data_editor(rows, key="angle_table", hide_index=True)

    dfs = [pd.DataFrame(f["rows"], columns=f["columns"]) for f in files]
    cols = list(dfs[0].columns)

    c1, c2 = st.columns(2)
    with c1:
        x_col = st.selectbox("X column", cols,
                             index=cols.index(s["x_col"]) if s.get("x_col") in cols
                             else min(1, len(cols) - 1))
    with c2:
        y_col = st.selectbox("Y column", cols,
                             index=cols.index(s["y_col"]) if s.get("y_col") in cols
                             else len(cols) - 1)

    xs_all = np.concatenate([pd.to_numeric(d[x_col], errors="coerce").dropna().values
                             for d in dfs])
    lo, hi = float(xs_all.min()), float(xs_all.max())
    st.caption(f"x range in files: {lo:g} to {hi:g}")

    x_text = st.text_input("X value(s), comma-separated for several",
                           value=s.get("x_val", f"{(lo + hi) / 2:g}"))
    slice_mode = st.radio("Slice mode", ["Linear interpolation", "Nearest row"],
                          index=0 if s.get("slice_mode") != "Nearest row" else 1,
                          horizontal=True)

    x_targets = [v for v in [t.strip() for t in x_text.split(",")] if v]
    try:
        vals = [float(v) for v in x_targets]
    except ValueError:
        st.error("X value(s) must be numbers, e.g. 5 or 2, 5, 8")
        return
    if len(vals) == 1:
        new = st.slider("X slider", min_value=lo, max_value=hi,
                        value=min(max(vals[0], lo), hi),
                        step=(hi - lo) / 400.0, format="%g")
        if abs(new - vals[0]) > 1e-12:
            vals = [new]

    c1, c2, c3 = st.columns(3)
    with c1:
        theta_zero = st.selectbox(
            "Theta zero", ["N (top)", "E (0 deg, standard)", "W", "S"],
            index=["N (top)", "E (0 deg, standard)", "W", "S"].index(
                s.get("theta_zero", "N (top)")))
        direction = st.selectbox("Direction", ["Counter-clockwise", "Clockwise"],
                                 index=0 if s.get("direction") != "Clockwise" else 1)
    with c2:
        marker = st.selectbox("Marker", ["o", "s", "^", "D", "x", "+", "*", "None"],
                              index=["o", "s", "^", "D", "x", "+", "*", "None"].index(
                                  s.get("marker", "o")))
        angle_display = st.selectbox(
            "Angle display", ["0 to 360", "-180 to 180", "-90 to 90"],
            index=["0 to 360", "-180 to 180", "-90 to 90"].index(
                s.get("angle_display", "0 to 360")))
    with c3:
        connect = st.checkbox("Connect points", value=s.get("connect", True))
        close = st.checkbox("Close circle", value=s.get("close", False))
        r_offset = st.checkbox("Offset R", value=s.get("r_offset", False))
        angle_labels = st.checkbox("Angle labels", value=s.get("angle_labels", True))

    title = st.text_input("Title", value=s.get("title", "Angular Dependence"))

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(7, 6), subplot_kw=dict(projection="polar"))
    ax.set_theta_zero_location(THETA_ZERO_MAP.get(theta_zero, "N"))
    ax.set_theta_direction(-1 if direction == "Clockwise" else 1)
    tick_degs = np.arange(0, 360, 30)
    ax.set_xticks(np.deg2rad(tick_degs))
    ax.set_xticklabels([f"{format_angle_deg(d, angle_display):g}\u00b0"
                        for d in tick_degs])

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    any_clamped, plotted = False, 0
    for k, x_target in enumerate(vals):
        pts, clamped = [], False
        for i, d in enumerate(dfs):
            if not bool(edited["include"].iloc[i]):
                continue
            theta = float(edited["angle"].iloc[i])
            xs = pd.to_numeric(d[x_col], errors="coerce").to_numpy(float)
            ys = pd.to_numeric(d[y_col], errors="coerce").to_numpy(float)
            m = np.isfinite(xs)
            if m.any() and (x_target < xs[m].min() or x_target > xs[m].max()):
                clamped = True
            try:
                r = value_at_x(xs, ys, x_target, slice_mode)
            except Exception:
                continue
            pts.append((theta, r))
        any_clamped = any_clamped or clamped
        if not pts:
            continue
        th = np.deg2rad([p[0] for p in pts])
        rs = np.array([p[1] for p in pts])
        if r_offset:
            rs = rs - rs.min()
        o = np.argsort(th)
        th, rs = th[o], rs[o]
        color = colors[k % len(colors)]
        mk = None if marker == "None" else marker
        if close:
            th = np.append(th, th[0] + 2 * np.pi)
            rs = np.append(rs, rs[0])
        ax.plot(th, rs, marker=mk, markersize=6, linewidth=2, color=color,
                linestyle="-" if connect else "",
                label=f"x = {x_target:g} @ {x_col}")
        if angle_labels:
            for t, r in zip(th, rs):
                ax.annotate(f"{format_angle_deg(np.rad2deg(t), angle_display):g}\u00b0",
                            xy=(t, r), fontsize=8, ha="center", va="bottom", color=color)
        plotted += 1

    if not plotted:
        st.warning("No points plotted (check angle / include / x settings).")
        plt.close(fig)
        return
    if len(vals) > 1:
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=9)
    ttl = title + (f"   (x = {vals[0]:g} @ {x_col})" if len(vals) == 1 else "")
    if any_clamped:
        ttl += "\nWARNING: x outside range for some file(s), clamped to end value"
    ax.set_title(ttl, fontsize=11, fontweight="bold", pad=20)
    st.pyplot(fig)
    plt.close(fig)


# ------------------------------------------------ Plot Data mode
LEGEND_POS = {"best": "best", "upper right": "upper right",
              "upper left": "upper left", "lower right": "lower right",
              "lower left": "lower left", "outside right": "center left"}


def _f(v, default=1.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def plot_data_view(session):
    datasets = session.get("datasets", {})
    if not datasets:
        st.warning("No datasets inside this session.")
        return
    sel = session.get("axis_selections", {})
    tl = session.get("titles_labels", {})
    ps = session.get("plot_settings", {})
    dt = session.get("data_transformation", {})
    ar = session.get("axis_ranges", {})
    gs = session.get("grid_settings", {})
    fs = session.get("font_settings", {})
    ls = session.get("legend_settings", {})
    ts = session.get("tick_settings", {})
    styles = session.get("styles", {})

    dfs = {name: pd.DataFrame(d["data"], columns=d["columns"])
           for name, d in datasets.items()}

    with st.sidebar:
        names = st.multiselect("Datasets", list(dfs), default=list(dfs))
        ref = dfs[names[0]] if names else next(iter(dfs.values()))
        cols = list(ref.columns)
        x_col = st.selectbox("X column", cols,
                             index=cols.index(sel["x_column"])
                             if sel.get("x_column") in cols else 0)
        default_y = [c for c in sel.get("y_column", []) if c in cols]
        y_cols = st.multiselect("Y columns", cols,
                                default=default_y or [cols[-1]])
        plot_type = st.selectbox("Plot type", ["Line", "Scatter"],
                                 index=1 if ps.get("plot_type",
                                                   session.get("plot_type")) == "Scatter" else 0)
        x_div = st.number_input("Divide X by", value=_f(dt.get("x_div")))
        y_div = st.number_input("Divide Y by", value=_f(dt.get("y_div")))
        color_mode = st.radio("Colors", ["Cycle", "Gradient"], horizontal=True)
        cmap_name = st.selectbox("Colormap",
                                 ["viridis", "plasma", "inferno", "magma", "turbo", "jet"],
                                 disabled=(color_mode != "Gradient"))
        c1, c2 = st.columns(2)
        x_log = c1.checkbox("X log", value=ps.get("x_log", False))
        y_log = c2.checkbox("Y log", value=ps.get("y_log", False))

        with st.expander("Axes & grid"):
            c1, c2 = st.columns(2)
            x_min = c1.number_input("X min", value=_f(ar.get("x_min"), np.nan),
                                    step=None, format="%g")
            x_max = c2.number_input("X max", value=_f(ar.get("x_max"), np.nan),
                                    step=None, format="%g")
            y_min = c1.number_input("Y min", value=_f(ar.get("y_min"), np.nan),
                                    step=None, format="%g")
            y_max = c2.number_input("Y max", value=_f(ar.get("y_max"), np.nan),
                                    step=None, format="%g")
            grid = st.checkbox("Major grid", value=gs.get("show_major_grid",
                                                          ps.get("show_grid", True)))
            grid_alpha = st.slider("Grid alpha", 0.0, 1.0,
                                   value=_f(gs.get("grid_alpha"), 0.3))
            grid_ls = st.selectbox("Grid linestyle", ["-", "--", ":", "-."],
                                   index=["-", "--", ":", "-."].index(
                                       gs.get("grid_linestyle", "-")
                                       if gs.get("grid_linestyle", "-") in
                                       ["-", "--", ":", "-."] else "-"))
            x_major = c1.number_input("X major tick (0=auto)",
                                      value=_f(ts.get("x_maj"), 0.0), min_value=0.0,
                                      format="%g")
            y_major = c2.number_input("Y major tick (0=auto)",
                                      value=_f(ts.get("y_maj"), 0.0), min_value=0.0,
                                      format="%g")

        with st.expander("Fonts & legend"):
            font_fam = st.selectbox("Font", ["Arial", "DejaVu Sans", "Calibri",
                                             "Times New Roman", "Consolas"])
            title_size = st.number_input("Title size", 8, 30,
                                         value=int(_f(fs.get("title_size"), 12)))
            label_size = st.number_input("Label size", 8, 30,
                                         value=int(_f(fs.get("label_size"), 11)))
            tick_size = st.number_input("Tick size", 6, 30,
                                        value=int(_f(fs.get("xtick_size"), 10)))
            leg_size = st.number_input("Legend size", 6, 30,
                                       value=int(_f(ls.get("legend_size"), 10)))
            leg_cols = st.number_input("Legend columns", 1, 6,
                                       value=int(_f(ls.get("columns"), 1)))
            show_legend = st.checkbox("Show legend",
                                      value=ls.get("show_legend", True))
            leg_pos = st.selectbox(
                "Legend position", list(LEGEND_POS),
                index=list(LEGEND_POS).index(
                    ls.get("position", "Best").lower().replace("_", " "))
                if ls.get("position", "Best").lower().replace("_", " ")
                in LEGEND_POS else 0)

    title = st.text_input("Title", value=tl.get("title", ""))
    c1, c2 = st.columns(2)
    xlabel = c1.text_input("X label", value=tl.get("xlabel", x_col))
    ylabel = c2.text_input("Y label", value=tl.get("ylabel", ""))

    if not names or not y_cols:
        st.warning("Pick datasets and Y columns.")
        return

    # ---- per-series style table (pre-filled from session styles) ----
    series = [(n, y) for n in names for y in y_cols if y in dfs[n].columns]
    st.subheader("Series styles (edit color / width / linestyle / label)")
    style_rows = []
    for n, y in series:
        stl = styles.get(f"{n}|||{y}", {})
        style_rows.append({
            "series": f"{n}: {y}",
            "label": stl.get("legend", f"{n}: {y}"),
            "color": stl.get("color", ""),
            "width": _f(stl.get("width"), 2.0),
            "linestyle": stl.get("linestyle", "-") or "-",
        })
    style_df = pd.DataFrame(style_rows)
    style_ed = st.data_editor(style_df, hide_index=True, key="style_table") \
        if len(style_rows) != 1 else style_df
    if len(style_rows) == 1:
        st.caption("single series - style via color mode / dividers above")

    fig, ax = plt.subplots(figsize=(9, 5))
    if color_mode == "Gradient" and series:
        cmap = plt.get_cmap(cmap_name)
        colors = [cmap(v) for v in np.linspace(0, 1, len(series))] \
            if len(series) > 1 else [cmap(0.5)]
    else:
        cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        colors = [cycle[i % len(cycle)] for i in range(len(series))]

    for i, ((n, y), c) in enumerate(zip(series, colors)):
        d = dfs[n]
        if x_col not in d.columns:
            continue
        row = style_ed.iloc[i]
        x = pd.to_numeric(d[x_col], errors="coerce").to_numpy(float) / x_div
        yy = pd.to_numeric(d[y], errors="coerce").to_numpy(float) / y_div
        color = row["color"] if isinstance(row["color"], str) and row["color"] else c
        lw = float(row["width"]) if row["width"] else 2.0
        lsty = row["linestyle"] if isinstance(row["linestyle"], str) and row["linestyle"] else "-"
        lbl = row["label"] if isinstance(row["label"], str) and row["label"] else f"{n}: {y}"
        if plot_type == "Scatter":
            ax.plot(x, yy, marker='o', markersize=5, linestyle='', color=color,
                    label=lbl)
        else:
            ax.plot(x, yy, color=color, linewidth=lw, linestyle=lsty, label=lbl)

    if x_log:
        ax.set_xscale("log")
    if y_log:
        ax.set_yscale("log")
    if grid:
        ax.grid(True, alpha=grid_alpha, linestyle=grid_ls)
    if x_major > 0:
        ax.xaxis.set_major_locator(plt.MultipleLocator(x_major))
    if y_major > 0:
        ax.yaxis.set_major_locator(plt.MultipleLocator(y_major))
    if not np.isnan(x_min) or not np.isnan(x_max):
        ax.set_xlim(None if np.isnan(x_min) else x_min,
                    None if np.isnan(x_max) else x_max)
    if not np.isnan(y_min) or not np.isnan(y_max):
        ax.set_ylim(None if np.isnan(y_min) else y_min,
                    None if np.isnan(y_max) else y_max)

    ax.set_xlabel(xlabel, fontsize=label_size, family=font_fam)
    ax.set_ylabel(ylabel, fontsize=label_size, family=font_fam)
    ax.set_title(title, fontsize=title_size, fontweight='bold', family=font_fam)
    ax.tick_params(labelsize=tick_size)
    if show_legend and series:
        ax.legend(fontsize=leg_size, loc=LEGEND_POS[leg_pos], ncol=int(leg_cols))

    st.pyplot(fig)
    plt.close(fig)
    st.caption(f"{len(series)} series from {len(names)} dataset(s)")


# ==================== main ====================
st.set_page_config(page_title="Session Plotter", layout="wide")
st.title("Session Plotter")
st.caption("Load a session JSON from 'Plot data.py' or 'angular polar.py'")

mode = st.sidebar.radio("Mode", ["Angular Polar", "Plot Data"])
up = st.sidebar.file_uploader("Session JSON", type="json")

session = None
if up is not None:
    try:
        session = json.load(up)
    except Exception as e:
        st.error(f"Bad JSON: {e}")
else:
    # optional: ?session=path/to/session.json in the URL
    qpath = st.query_params.get("session")
    if qpath and os.path.isfile(qpath):
        try:
            with open(qpath, "r", encoding="utf-8") as f:
                session = json.load(f)
        except Exception as e:
            st.error(f"Bad session file {qpath}: {e}")

if session is not None:
        if mode == "Angular Polar":
            if "files" not in session:
                st.error("No 'files' key - looks like a Plot Data session. Switch mode.")
            else:
                angular_polar_view(session)
        else:
            if "datasets" not in session:
                st.error("No 'datasets' key - looks like an Angular Polar session. Switch mode.")
            else:
                plot_data_view(session)
else:
    st.info("Upload a session JSON in the sidebar, pick the mode, and the plot appears here.")

