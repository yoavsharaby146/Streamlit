"""Streamlit combo viewer: Plot Data sessions + Angular Polar sessions.

Loads the JSON session files written by 'Plot data.py' and
'angular polar.py' and reproduces their core plotting features.
Several session files can be loaded together (sidebar multi-upload, or
?session=path/a.json;path/b.json in the URL) and shown side by side
for visual comparison, with optional shared settings synced across
all panels.

Run:  streamlit run streamlit_app.py
"""
import hashlib
import io
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==================== shared helpers (same logic as the Tk apps) ====================
ANGLE_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:deg|°)", re.IGNORECASE)
ANY_NUM_RE = re.compile(r"(-?\d+(?:\.\d+)?)")

THETA_ZERO_OPTS = ["N (top)", "E (0 deg, standard)", "W", "S"]
THETA_ZERO_MAP = {"E (0 deg, standard)": "E", "N (top)": "N", "W": "W", "S": "S"}
DIRECTION_OPTS = ["Counter-clockwise", "Clockwise"]
MARKER_OPTS = ["o", "s", "^", "D", "x", "+", "*", "None"]
ANGLE_DISPLAY_OPTS = ["0 to 360", "-180 to 180", "-90 to 90"]
SLICE_MODES = ["Linear interpolation", "Nearest row"]
LEGEND_POS = {"best": "best", "upper right": "upper right",
              "upper left": "upper left", "lower right": "lower right",
              "lower left": "lower left", "outside right": "center left"}
# every colormap matplotlib knows (used by Plot Data "Gradient" colors)
CMAPS = sorted(matplotlib.colormaps())
TYPE_LABEL = {"angular": "Angular Polar", "plotdata": "Plot Data"}


# fonts actually installed where the app runs (Streamlit Cloud's Linux
# container has no Arial/Calibri etc. - only list what matplotlib finds)
FONT_OPTIONS = sorted({f.name for f in matplotlib.font_manager.fontManager.ttflist})
DEFAULT_FONT = "DejaVu Sans" if "DejaVu Sans" in FONT_OPTIONS else (FONT_OPTIONS[0] if FONT_OPTIONS else "")


def _f(v, default=1.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def opt_index(opts, value, default=0):
    """Index of value in opts (safe against values saved by other apps)."""
    return opts.index(value) if value in opts else default


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


def detect_session_type(session):
    """'angular' | 'plotdata' | None, from the session's top-level keys."""
    if not isinstance(session, dict):
        return None
    if "files" in session:
        return "angular"
    if "datasets" in session:
        return "plotdata"
    return None


def _grid_map(xs, ys, zs, res=300):
    """Grid (x, y, z) data onto a regular grid for Color Map plots.

    Regular measurement grids (every x*y combination present, the common
    case for sweep maps) are reshaped exactly with no interpolation;
    scattered data is linearly interpolated (scipy when available, nearest
    bucketing otherwise). NaN cells stay transparent in the plot.
    Returns (zi, x_centers, y_centers).
    """
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    zs = np.asarray(zs, float)
    m = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs)
    xs, ys, zs = xs[m], ys[m], zs[m]
    if xs.size == 0:
        return np.array([[np.nan]]), np.array([0.0]), np.array([0.0])
    xu, xi = np.unique(xs, return_inverse=True)
    yu, yi = np.unique(ys, return_inverse=True)
    if len(xu) * len(yu) == xs.size:
        zi = np.full((len(yu), len(xu)), np.nan)
        zi[yi, xi] = zs
        return zi, xu, yu
    gx = np.linspace(xs.min(), xs.max(), res)
    gy = np.linspace(ys.min(), ys.max(), res)
    try:
        from scipy.interpolate import griddata
        mx, my = np.meshgrid(gx, gy)
        return griddata((xs, ys), zs, (mx, my), method="linear"), gx, gy
    except ImportError:
        pass
    # numpy fallback: bucket each point into the nearest grid cell
    zi = np.full((len(gy), len(gx)), np.nan)
    ix = np.clip(np.searchsorted(gx, xs) - 1, 0, len(gx) - 1)
    iy = np.clip(np.searchsorted(gy, ys) - 1, 0, len(gy) - 1)
    zi[iy, ix] = zs
    return zi, gx, gy


if __name__ == "__main__" and "--self-check" in sys.argv:
    # pure helpers only - no widgets, so this runs headless
    assert detect_session_type({"files": []}) == "angular"
    assert detect_session_type({"datasets": {}}) == "plotdata"
    assert detect_session_type({"foo": 1}) is None
    assert detect_session_type(["not", "a", "dict"]) is None
    assert value_at_x([0, 10, 20, 30], [1, 3, 5, 7], 15) == 4.0
    assert value_at_x([30, 0, 20, 10], [7, 1, 5, 3], 15) == 4.0
    assert value_at_x([0, 10, 20], [1, 3, 5], 19, "Nearest row") == 5
    assert format_angle_deg(270, "-180 to 180") == -90.0
    assert format_angle_deg(135, "-90 to 90") == -45.0
    assert detect_angle_from_name("sample_30deg.csv") == 30.0
    assert detect_angle(r"C:\data\30 deg\75deg.csv") == 30.0
    assert detect_angle(r"C:\data\runs\75deg.csv") == 75.0
    assert opt_index(["a", "b"], "b") == 1 and opt_index(["a"], "z") == 0
    # color-map gridding: regular grid reshapes exactly, values land in place
    zi, gx, gy = _grid_map(np.array([0., 1., 0., 1.]), np.array([0., 0., 1., 1.]),
                           np.array([1., 2., 3., 4.]))
    assert zi.shape == (2, 2) and zi[0, 0] == 1.0 and zi[0, 1] == 2.0 \
        and zi[1, 0] == 3.0 and zi[1, 1] == 4.0, zi
    print("streamlit_app self-check OK")
    sys.exit(0)

# ==================== session loading ====================
def collect_sessions():
    """Gather (name, type, session) panels from multi-upload + ?session= URLs."""
    raw = []  # (name, parsed-or-None)
    ups = st.sidebar.file_uploader("Session JSON(s)", type="json",
                                   accept_multiple_files=True)
    for f in ups or []:
        try:
            raw.append((f.name, json.load(f)))
        except Exception as e:
            st.sidebar.error(f"{f.name}: bad JSON ({e})")
    # ?session=a.json;b.json in the URL (each entry may itself be a
    # ';'-separated list, and the param may be repeated)
    for qval in st.query_params.get_all("session"):
        for q in (p.strip() for p in qval.split(";")):
            if not q:
                continue
            if not os.path.isfile(q):
                st.sidebar.warning(f"session file not found: {q}")
                continue
            try:
                with open(q, "r", encoding="utf-8") as fh:
                    raw.append((os.path.basename(q), json.load(fh)))
            except Exception as e:
                st.sidebar.error(f"{q}: bad session ({e})")

    panels = []
    for name, s in raw:
        t = detect_session_type(s)
        if t is None:
            st.sidebar.error(f"{name}: unknown session format "
                             "(needs a 'files' or 'datasets' key)")
        else:
            panels.append({"name": name, "type": t, "session": s})
    return panels


def shared_settings():
    """One control block whose values are applied to every panel.
    Blank text fields mean 'keep each panel's own choice'."""
    with st.expander("Shared settings — applied to every panel", expanded=True):
        c1, c2, c3 = st.columns(3)
        x_col = c1.text_input("X column", key="shared-xcol",
                              help="Applied where the column exists; blank = per panel")
        y_col = c2.text_input("Y column", key="shared-ycol",
                              help="First Y for Plot Data; blank = per panel")
        title = c3.text_input("Title", key="shared-title",
                              help="Overrides every panel title; blank = per panel")
        x_text = c1.text_input("X value(s) — polar slices", key="shared-xtext",
                               help="e.g. 5 or 2, 5, 8; blank = per panel")
        slice_mode = c2.selectbox("Polar slice mode", SLICE_MODES)
        theta_zero = c3.selectbox("Polar theta zero", THETA_ZERO_OPTS)
        c4, c5, c6 = st.columns(3)
        direction = c4.selectbox("Polar direction", DIRECTION_OPTS)
        angle_display = c5.selectbox("Angle display", ANGLE_DISPLAY_OPTS)
        plot_type = c6.selectbox("Plot type (Plot Data)",
                                 ["Line", "Scatter", "Color Map"])
        c7, c8, c9, c10 = st.columns(4)
        x_div = c7.number_input("Divide X by", value=1.0, min_value=0.0)
        y_div = c8.number_input("Divide Y by", value=1.0, min_value=0.0)
        x_log = c9.checkbox("X log")
        y_log = c10.checkbox("Y log")
        grid = st.checkbox("Grid on (Plot Data)", value=True)
    return {"x_col": x_col or None, "y_col": y_col or None,
            "title": title or None, "x_text": x_text or None,
            "slice_mode": slice_mode, "theta_zero": theta_zero,
            "direction": direction, "angle_display": angle_display,
            "plot_type": plot_type, "x_div": x_div or None, "y_div": y_div or None,
            "x_log": x_log, "y_log": y_log, "grid": grid}


def download_png(fig, pid):
    """Small per-panel PNG export button."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    st.download_button("Download PNG", buf.getvalue(), file_name=f"{pid}.png",
                       mime="image/png", key=f"{pid}-dl")


def show_fig(fig, dpi=110):
    """Render a figure at its natural pixel size (figsize x dpi).

    Unlike width='stretch', the on-screen size stays the same whether
    sidebars / expanders are open or closed, so side-by-side panels keep
    their relative sizes; it only shrinks if the column is narrower.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    st.image(buf.getvalue())

def _x_slider_changed(pid):
    """A panel's X slider moved: write the value back to its source —
    the shared X field when syncing (all panels scrub together), or the
    panel's own X field otherwise."""
    val = st.session_state[f"{pid}-xslider"]
    text = f"{val:g}"
    st.session_state[f"{pid}-xtext_last"] = text
    if st.session_state.get("sync-panels", False):
        st.session_state["shared-xtext"] = text
    else:
        st.session_state[f"{pid}-xval"] = text


def _z_col_changed(pid):
    """Color Map Z column changed: swap in that column's remembered
    colorbar label and Z color range, so switching Z is smooth."""
    zc = st.session_state[f"{pid}-zcol"]
    p = st.session_state.get(f"{pid}-zprefs-{zc}") or {}
    st.session_state[f"{pid}-zlabel"] = p.get("label", zc)
    st.session_state[f"{pid}-zmin"] = p.get("zmin", np.nan)
    st.session_state[f"{pid}-zmax"] = p.get("zmax", np.nan)


def _x_col_changed_sw(pid):
    """Plot Data X column changed: swap in that column's remembered axis
    label and X range (per-column sticky memory, like the Z feature)."""
    xc = st.session_state[f"{pid}-xcol"]
    p = st.session_state.get(f"{pid}-xprefs-{xc}") or {}
    st.session_state[f"{pid}-xlabel"] = p.get("label", xc)
    st.session_state[f"{pid}-xmin"] = p.get("xmin", np.nan)
    st.session_state[f"{pid}-xmax"] = p.get("xmax", np.nan)


def _y_col_changed_sw(pid):
    """Plot Data Y selection changed: swap the FIRST selected Y column's
    remembered axis label and Y range into the widgets."""
    ycs = st.session_state.get(f"{pid}-ycols") or []
    new = ycs[0] if ycs else None
    old = st.session_state.get(f"{pid}-ycol_cur")
    if new == old:
        return
    if old:
        st.session_state[f"{pid}-yprefs-{old}"] = {
            "label": st.session_state.get(f"{pid}-ylabel", ""),
            "ymin": st.session_state.get(f"{pid}-ymin", np.nan),
            "ymax": st.session_state.get(f"{pid}-ymax", np.nan),
        }
    p = (st.session_state.get(f"{pid}-yprefs-{new}") or {}) if new else {}
    st.session_state[f"{pid}-ylabel"] = p.get("label", new or "")
    st.session_state[f"{pid}-ymin"] = p.get("ymin", np.nan)
    st.session_state[f"{pid}-ymax"] = p.get("ymax", np.nan)
    st.session_state[f"{pid}-ycol_cur"] = new


# ==================== Angular Polar panel ====================
def render_angular(session, pid, ov, compact):
    files = session.get("files", [])
    if not files:
        st.warning("No files inside this session.")
        return
    s = session.get("settings", {})

    # the plot renders FIRST in the panel (settings go in collapsible
    # expanders below it) so panels align side by side regardless of
    # how many settings they have
    plot_box = st.container()

    def fname(p):
        return p.replace("\\", "/").rsplit("/", 1)[-1]

    rows = pd.DataFrame({
        "include": [bool(f.get("keep", True)) for f in files],
        "angle": [float(f.get("angle") or detect_angle(f.get("path", "")) or 0)
                  for f in files],
        "file": [fname(f.get("path", "?")) for f in files],
    })
    with st.expander("Files & angles (edit angle / untick include)",
                     expanded=not compact):
        edited = st.data_editor(rows, hide_index=True, key=f"{pid}-angle_table")

    dfs = [pd.DataFrame(f["rows"], columns=f["columns"]) for f in files]
    cols = list(dfs[0].columns)

    def pick(shared, saved, fallback):
        for v in (shared, saved):
            if v in cols:
                return v
        return fallback

    if ov is not None:
        # synced: shared values override; missing columns fall back per panel
        x_col = pick(ov["x_col"], s.get("x_col"), cols[min(1, len(cols) - 1)])
        y_col = pick(ov["y_col"], s.get("y_col"), cols[-1])
        x_text = ov["x_text"] or s.get("x_val", "")
        slice_mode = ov["slice_mode"]
        theta_zero = ov["theta_zero"]
        direction = ov["direction"]
        angle_display = ov["angle_display"]
        marker = s.get("marker", "o") if s.get("marker") in MARKER_OPTS else "o"
        connect = s.get("connect", True)
        close = s.get("close", False)
        r_offset = s.get("r_offset", False)
        angle_labels = s.get("angle_labels", True)
        title = ov["title"] or s.get("title", "Angular Dependence")
    else:
        with st.expander("Polar settings", expanded=not compact):
            c1, c2 = st.columns(2)
            x_col = c1.selectbox(
                "X column", cols, key=f"{pid}-xcol",
                index=opt_index(cols, s.get("x_col"), min(1, len(cols) - 1)))
            y_col = c2.selectbox(
                "Y column", cols, key=f"{pid}-ycol",
                index=opt_index(cols, s.get("y_col"), len(cols) - 1))
            x_text = st.text_input("X value(s), comma-separated for several",
                                   value=s.get("x_val", ""), key=f"{pid}-xval")
            slice_mode = st.radio("Slice mode", SLICE_MODES, horizontal=True,
                                  key=f"{pid}-slice",
                                  index=opt_index(SLICE_MODES, s.get("slice_mode")))
            c1, c2, c3 = st.columns(3)
            theta_zero = c1.selectbox("Theta zero", THETA_ZERO_OPTS, key=f"{pid}-tz",
                                      index=opt_index(THETA_ZERO_OPTS, s.get("theta_zero")))
            direction = c2.selectbox("Direction", DIRECTION_OPTS, key=f"{pid}-dir",
                                     index=opt_index(DIRECTION_OPTS, s.get("direction")))
            marker = c3.selectbox("Marker", MARKER_OPTS, key=f"{pid}-mk",
                                  index=opt_index(MARKER_OPTS, s.get("marker")))
            angle_display = c1.selectbox("Angle display", ANGLE_DISPLAY_OPTS,
                                         key=f"{pid}-adisp",
                                         index=opt_index(ANGLE_DISPLAY_OPTS,
                                                         s.get("angle_display")))
            connect = c2.checkbox("Connect", value=s.get("connect", True),
                                  key=f"{pid}-conn")
            close = c3.checkbox("Close circle", value=s.get("close", False),
                                key=f"{pid}-close")
            r_offset = c1.checkbox("Offset R", value=s.get("r_offset", False),
                                   key=f"{pid}-roff")
            angle_labels = c2.checkbox("Angle labels",
                                       value=s.get("angle_labels", True),
                                       key=f"{pid}-alab")
            title = st.text_input("Title", value=s.get("title", "Angular Dependence"),
                                  key=f"{pid}-title")

    xs_all = np.concatenate([pd.to_numeric(d[x_col], errors="coerce").dropna().values
                             for d in dfs]) if dfs else np.array([0.0, 1.0])
    if xs_all.size == 0:
        xs_all = np.array([0.0, 1.0])
    lo, hi = float(xs_all.min()), float(xs_all.max())

    x_text = x_text or f"{(lo + hi) / 2:g}"
    x_targets = [v for v in (t.strip() for t in x_text.split(",")) if v]
    try:
        vals = [float(v) for v in x_targets]
    except ValueError:
        st.error("X value(s) must be numbers, e.g. 5 or 2, 5, 8")
        return
    # keep the panel slider in sync with the typed / shared X value
    # (the slider itself is rendered under the plot, below)
    if len(vals) == 1 and hi > lo:
        if st.session_state.get(f"{pid}-xtext_last") != f"{vals[0]:g}":
            st.session_state[f"{pid}-xslider"] = min(max(vals[0], lo), hi)
            st.session_state[f"{pid}-xtext_last"] = f"{vals[0]:g}"

    # ---- plot ----
    # compact (side-by-side): shorter figure so the slider + caption
    # below still keep the panel within one screen height
    fig, ax = plt.subplots(figsize=(6.2, 4.5) if compact else (7, 6),
                           subplot_kw=dict(projection="polar"))
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
        with plot_box:
            st.warning("No points plotted (check angle / include / x settings).")
        plt.close(fig)
        return
    if len(vals) > 1:
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=9)
    ttl = title + (f"   (x = {vals[0]:g} @ {x_col})" if len(vals) == 1 else "")
    if any_clamped:
        ttl += "\nWARNING: x outside range for some file(s), clamped to end value"
    ax.set_title(ttl, fontsize=11, fontweight="bold", pad=20)
    with plot_box:
        show_fig(fig)
        st.caption(f"x range in files ({x_col}): {lo:g} to {hi:g}")
        # X slice slider under the plot: dragging it updates the shared X
        # when syncing (all panels scrub together), else only this panel
        if len(vals) == 1 and hi > lo:
            st.slider("X slice", min_value=lo, max_value=hi,
                      step=(hi - lo) / 400.0, format="%g", key=f"{pid}-xslider",
                      on_change=_x_slider_changed, args=(pid,))
        download_png(fig, pid)
    plt.close(fig)


# ==================== Plot Data panel ====================
def render_plotdata(session, pid, ov, compact):
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

    # the plot renders FIRST in the panel; all settings live in
    # collapsible expanders below it so panels align side by side
    plot_box = st.container()

    # ---- data & transform ----
    with st.expander("Data & transform", expanded=not compact):
        names = st.multiselect("Datasets", list(dfs), default=list(dfs),
                               key=f"{pid}-names")
        ref = dfs[names[0]] if names else next(iter(dfs.values()))
        cols = list(ref.columns)
        x_col = (ov["x_col"] if ov and ov["x_col"] in cols else
                 st.selectbox("X column", cols, key=f"{pid}-xcol",
                              index=opt_index(cols, sel.get("x_column"), 0),
                              on_change=_x_col_changed_sw, args=(pid,)))
        # per-X sticky settings: each X column remembers its own axis label
        # and X range; switching X swaps them in (see _x_col_changed_sw)
        xpref_key = f"{pid}-xprefs-{x_col}"
        if st.session_state.get(xpref_key) is None:
            x0 = sel.get("x_column")
            st.session_state[xpref_key] = {
                "label": tl.get("xlabel", x_col) if x_col == x0 else x_col,
                "xmin": _f(ar.get("x_min"), np.nan) if x_col == x0 else np.nan,
                "xmax": _f(ar.get("x_max"), np.nan) if x_col == x0 else np.nan,
            }
        for _wk, _pk in ((f"{pid}-xlabel", "label"), (f"{pid}-xmin", "xmin"),
                         (f"{pid}-xmax", "xmax")):
            if _wk not in st.session_state:
                st.session_state[_wk] = st.session_state[xpref_key][_pk]
        default_y = [c for c in sel.get("y_column", []) if c in cols]
        if ov and ov["y_col"] in cols:
            y_cols = [ov["y_col"]]
        else:
            y_cols = st.multiselect("Y columns", cols,
                                    default=default_y or [cols[-1]],
                                    key=f"{pid}-ycols",
                                    on_change=_y_col_changed_sw, args=(pid,))
        # per-Y sticky settings: the FIRST selected Y column drives the axis
        if f"{pid}-ycol_cur" not in st.session_state:
            st.session_state[f"{pid}-ycol_cur"] = y_cols[0] if y_cols else None
        for _wk, _v in ((f"{pid}-ylabel", tl.get("ylabel", "")),
                        (f"{pid}-ymin", _f(ar.get("y_min"), np.nan)),
                        (f"{pid}-ymax", _f(ar.get("y_max"), np.nan))):
            if _wk not in st.session_state:
                st.session_state[_wk] = _v
        plot_type = (ov["plot_type"] if ov else None) or st.selectbox(
            "Plot type", ["Line", "Scatter", "Color Map"], key=f"{pid}-ptype",
            index=opt_index(["Line", "Scatter", "Color Map"],
                            ps.get("plot_type", session.get("plot_type"))))
        x_div = (ov["x_div"] if ov and ov["x_div"] else None) or \
            st.number_input("Divide X by", value=_f(dt.get("x_div")), key=f"{pid}-xdiv")
        y_div = (ov["y_div"] if ov and ov["y_div"] else None) or \
            st.number_input("Divide Y by", value=_f(dt.get("y_div")), key=f"{pid}-ydiv")
        if plot_type == "Color Map":
            # Color Map: Z column drives the color, first Y column is the map's Y axis
            if len(y_cols) > 1:
                y_cols = y_cols[:1]
                st.caption("Color Map uses the first Y column as the map's Y axis")
            z_col = st.selectbox(
                "Z column (color)", cols, key=f"{pid}-zcol",
                index=opt_index(cols, sel.get("z_column"), len(cols) - 1),
                on_change=_z_col_changed, args=(pid,))
            # per-Z sticky settings: every Z column remembers its own
            # colorbar label and color range; switching Z swaps them in
            zpref_key = f"{pid}-zprefs-{z_col}"
            if st.session_state.get(zpref_key) is None:
                z0 = sel.get("z_column")
                st.session_state[zpref_key] = {
                    "label": tl.get("zlabel", z_col) if z_col == z0 else z_col,
                    "zmin": _f(ar.get("z_min"), np.nan) if z_col == z0 else np.nan,
                    "zmax": _f(ar.get("z_max"), np.nan) if z_col == z0 else np.nan,
                }
            for _wk, _pk in ((f"{pid}-zlabel", "label"), (f"{pid}-zmin", "zmin"),
                             (f"{pid}-zmax", "zmax")):
                if _wk not in st.session_state:
                    st.session_state[_wk] = st.session_state[zpref_key][_pk]
            z_div = st.number_input("Divide Z by", value=_f(dt.get("z_div")),
                                    key=f"{pid}-zdiv")
        color_mode = st.radio("Colors", ["Cycle", "Gradient"], horizontal=True,
                              key=f"{pid}-cmode")
        cmap_name = st.selectbox(
            "Colormap", CMAPS, key=f"{pid}-cmap",
            index=opt_index(CMAPS, ps.get("colormap", session.get("colormap")),
                            CMAPS.index("viridis")),
            disabled=(color_mode != "Gradient" and plot_type != "Color Map"))
        if ov is not None:
            st.caption(f"synced: {plot_type}, divide X by {x_div:g}, "
                       f"Y by {y_div:g}, X log={ov['x_log']}, Y log={ov['y_log']}")
        else:
            c1, c2 = st.columns(2)
            x_log = c1.checkbox("X log", value=ps.get("x_log", False), key=f"{pid}-xlog")
            y_log = c2.checkbox("Y log", value=ps.get("y_log", False), key=f"{pid}-ylog")
    if ov is not None:
        x_log, y_log = ov["x_log"], ov["y_log"]

    # ---- axes, grid & ticks ----
    with st.expander("Axes, grid & ticks"):
        c1, c2 = st.columns(2)
        # defaults come from the per-column preferences in session state
        x_min = c1.number_input("X min", step=None, format="%g",
                                key=f"{pid}-xmin")
        x_max = c2.number_input("X max", step=None, format="%g",
                                key=f"{pid}-xmax")
        y_min = c1.number_input("Y min", step=None, format="%g",
                                key=f"{pid}-ymin")
        y_max = c2.number_input("Y max", step=None, format="%g",
                                key=f"{pid}-ymax")
        if plot_type == "Color Map":
            # defaults come from the per-Z preferences in session state
            z_min = c1.number_input("Z min (color)", step=None, format="%g",
                                    key=f"{pid}-zmin")
            z_max = c2.number_input("Z max (color)", step=None, format="%g",
                                    key=f"{pid}-zmax")
        grid = (ov["grid"] if ov is not None else
                st.checkbox("Major grid",
                            value=gs.get("show_major_grid", ps.get("show_grid", True)),
                            key=f"{pid}-grid"))
        grid_alpha = st.slider("Grid alpha", 0.0, 1.0,
                               value=_f(gs.get("grid_alpha"), 0.3), key=f"{pid}-galpha")
        grid_ls = st.selectbox("Grid linestyle", ["-", "--", ":", "-."],
                               key=f"{pid}-gls",
                               index=opt_index(["-", "--", ":", "-."],
                                               gs.get("grid_linestyle", "-")))
        x_major = c1.number_input("X major tick (0=auto)",
                                  value=_f(ts.get("x_maj"), 0.0), min_value=0.0,
                                  format="%g", key=f"{pid}-xmaj")
        y_major = c2.number_input("Y major tick (0=auto)",
                                  value=_f(ts.get("y_maj"), 0.0), min_value=0.0,
                                  format="%g", key=f"{pid}-ymaj")

    # ---- fonts & legend ----
    with st.expander("Fonts & legend"):
        font_fam = st.selectbox("Font", FONT_OPTIONS, key=f"{pid}-font",
                                index=opt_index(FONT_OPTIONS, DEFAULT_FONT))
        title_size = st.number_input("Title size", 8, 30,
                                     value=int(_f(fs.get("title_size"), 12)),
                                     key=f"{pid}-tsize")
        label_size = st.number_input("Label size", 8, 30,
                                     value=int(_f(fs.get("label_size"), 11)),
                                     key=f"{pid}-lsize")
        tick_size = st.number_input("Tick size", 6, 30,
                                    value=int(_f(fs.get("xtick_size"), 10)),
                                    key=f"{pid}-tick")
        leg_size = st.number_input("Legend size", 6, 30,
                                   value=int(_f(ls.get("legend_size"), 10)),
                                   key=f"{pid}-legsize")
        leg_cols = st.number_input("Legend columns", 1, 6,
                                   value=int(_f(ls.get("columns"), 1)),
                                   key=f"{pid}-legcols")
        show_legend = st.checkbox("Show legend", value=ls.get("show_legend", True),
                                  key=f"{pid}-legshow")
        leg_pos = st.selectbox(
            "Legend position", list(LEGEND_POS), key=f"{pid}-legpos",
            index=opt_index(list(LEGEND_POS),
                            ls.get("position", "Best").lower().replace("_", " ")))

    with st.expander("Title & axis labels", expanded=not compact):
        title = (ov["title"] if ov and ov["title"] else None) or \
            st.text_input("Title", value=tl.get("title", ""), key=f"{pid}-title")
        c1, c2 = st.columns(2)
        xlabel = c1.text_input("X label", key=f"{pid}-xlabel")
        ylabel = c2.text_input("Y label", key=f"{pid}-ylabel")
        if plot_type == "Color Map":
            zlabel = st.text_input("Z label (colorbar)", key=f"{pid}-zlabel")

    # sticky memory: remember the current X / first-Y column's label & range
    st.session_state[xpref_key] = {
        "label": st.session_state.get(f"{pid}-xlabel", x_col),
        "xmin": st.session_state.get(f"{pid}-xmin", np.nan),
        "xmax": st.session_state.get(f"{pid}-xmax", np.nan),
    }
    if y_cols:
        _yk = y_cols[0]
        st.session_state[f"{pid}-yprefs-{_yk}"] = {
            "label": st.session_state.get(f"{pid}-ylabel", ""),
            "ymin": st.session_state.get(f"{pid}-ymin", np.nan),
            "ymax": st.session_state.get(f"{pid}-ymax", np.nan),
        }

    if not names or not y_cols:
        st.warning("Pick datasets and Y columns.")
        return

    # ---- Color Map: 2D map of the Z column in color (like 'Plot data.py') ----
    if plot_type == "Color Map":
        y_col = y_cols[0]
        # remember this Z column's current label / color range so that
        # switching Z and coming back restores them
        st.session_state[zpref_key] = {
            "label": st.session_state.get(f"{pid}-zlabel", z_col),
            "zmin": st.session_state.get(f"{pid}-zmin", np.nan),
            "zmax": st.session_state.get(f"{pid}-zmax", np.nan),
        }
        fig, ax = plt.subplots(figsize=(6.2, 5.1) if compact else (9, 5))
        maps, zvals = [], []
        for n in names:
            d = dfs[n]
            if not all(c in d.columns for c in (x_col, y_col, z_col)):
                continue
            xs = pd.to_numeric(d[x_col], errors="coerce").to_numpy(float) / x_div
            ys = pd.to_numeric(d[y_col], errors="coerce").to_numpy(float) / y_div
            zs = pd.to_numeric(d[z_col], errors="coerce").to_numpy(float) / z_div
            zi, gx, gy = _grid_map(xs, ys, zs)
            maps.append((zi, gx, gy))
            zvals.append(zs[np.isfinite(zs)])
        if not maps:
            st.warning("Color Map: X/Y/Z columns not found in the selected datasets.")
            plt.close(fig)
            return
        zf = np.concatenate(zvals) if zvals else np.array([np.nan])
        vmin = None if np.isnan(z_min) else z_min
        vmax = None if np.isnan(z_max) else z_max
        if np.isfinite(zf).any():
            if vmin is None:
                vmin = float(np.min(zf))
            if vmax is None:
                vmax = float(np.max(zf))
        if vmin is not None and vmax is not None and vmax < vmin:
            vmin, vmax = None, None  # invalid manual range -> auto
        im = None
        for zi, gx, gy in maps:
            im = ax.pcolormesh(gx, gy, zi, shading="nearest",
                               cmap=plt.get_cmap(cmap_name), vmin=vmin, vmax=vmax)
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
        if grid:
            ax.grid(True, alpha=grid_alpha, linestyle=grid_ls,
                    color="white", linewidth=0.6)
        ax.set_xlabel(xlabel, fontsize=label_size, family=font_fam)
        ax.set_ylabel(ylabel, fontsize=label_size, family=font_fam)
        ax.set_title(title, fontsize=title_size, fontweight='bold', family=font_fam)
        ax.tick_params(labelsize=tick_size)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(zlabel, fontsize=label_size, family=font_fam)
        cbar.ax.tick_params(labelsize=tick_size)
        fig.tight_layout()
        with plot_box:
            show_fig(fig)
            download_png(fig, pid)
            st.caption(f"{len(maps)} color map(s) from {len(names)} dataset(s)")
        plt.close(fig)
        return

    # ---- per-series style table (pre-filled from session styles) ----
    series = [(n, y) for n in names for y in y_cols if y in dfs[n].columns]
    with st.expander("Series styles (edit color / width / linestyle / label)",
                     expanded=not compact):
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
        # key includes a signature of the series list: a fresh key when the
        # series selection changes avoids stale data_editor state
        sig = hashlib.md5("|".join(f"{n}|||{y}" for n, y in series).encode()).hexdigest()[:8]
        style_ed = st.data_editor(pd.DataFrame(style_rows), hide_index=True,
                                  key=f"{pid}-style-{sig}")

    # compact: taller than the polar figure on purpose - the polar panel
    # carries its slider + caption below the plot, so both panel types
    # end up roughly the same overall height side by side
    fig, ax = plt.subplots(figsize=(6.2, 5.1) if compact else (9, 5))
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

    with plot_box:
        show_fig(fig)
        download_png(fig, pid)
        st.caption(f"{len(series)} series from {len(names)} dataset(s)")
    plt.close(fig)

# ==================== main ====================
st.set_page_config(page_title="Session Plotter", layout="wide")
st.title("Session Plotter")
st.caption("Load session JSONs from 'Plot data.py' / 'angular polar.py' "
           "(multi-upload, or ?session=a.json;b.json) and compare side by side.")

panels = collect_sessions()
# shows which deployed commit is running (Streamlit Cloud sets this)
_commit = os.environ.get("STREAMLIT_GIT_COMMIT_HASH")
if _commit:
    st.sidebar.caption(f"deployed commit: {_commit[:7]}")
if not panels:
    st.info("Upload one or more session JSONs in the sidebar and the plots "
            "appear here. Sessions from both Tk apps are detected "
            "automatically ('files' = Angular Polar, 'datasets' = Plot Data).")
    st.stop()

top1, top2 = st.columns([1, 1])
# ?cols=1|2|3 (or ?per_row=) forces the layout for a shared link;
# without it: 1 session -> full width, 2+ -> two per row
url_cols = None
for q in st.query_params.get_all("cols") + st.query_params.get_all("per_row"):
    try:
        v = int(q)
    except (TypeError, ValueError):
        continue
    if v in (1, 2, 3):
        url_cols = v
        break
default_cols = url_cols or min(len(panels), 2)
ncols = top1.selectbox("Panels per row", [1, 2, 3],
                       index=[1, 2, 3].index(default_cols), key="panels-per-row",
                       help="Can also be set in the URL, e.g. &cols=3")
sync = len(panels) > 1 and top2.toggle("Sync settings across panels", value=True,
                                       key="sync-panels")
st.caption(f"{len(panels)} session(s): " + " · ".join(
    f"`{p['name']}` ({TYPE_LABEL[p['type']]})" for p in panels))

ov = shared_settings() if sync else None

# grid of panels: each uploaded/URL session becomes one plot panel
for r0 in range(0, len(panels), ncols):
    row_panels = panels[r0:r0 + ncols]
    grid = st.columns(ncols)
    for j, (col, p) in enumerate(zip(grid, row_panels)):
        pid = f"panel{r0 + j}"
        with col:
            st.markdown(f"**{p['name']}** — `{TYPE_LABEL[p['type']]}`")
            st.divider()
            try:
                if p["type"] == "angular":
                    render_angular(p["session"], pid, ov, compact=ncols > 1)
                else:
                    render_plotdata(p["session"], pid, ov, compact=ncols > 1)
            except Exception as e:  # one bad session never kills the page
                st.error(f"{p['name']}: {e}")
