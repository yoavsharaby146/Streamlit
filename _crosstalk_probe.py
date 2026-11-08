"""Cross-talk probe: change every widget on one panel and verify the other
panels' widget values are untouched after each change."""
import hashlib
import json as _json
import os as _os
import tempfile as _tf

from streamlit.testing.v1 import AppTest

APP = r"c:\Users\Yoav\VScode\streamlit-plotter\streamlit_app.py"
pid_for = lambda n: "p-" + hashlib.md5(n.encode("utf-8")).hexdigest()[:6]


def write(name, obj):
    p = _os.path.join(_tf.mkdtemp(), name)
    with open(p, "w") as fh:
        _json.dump(obj, fh)
    return p


ANG = {"files": [
    {"path": "a_0deg.csv", "angle": 0.0, "columns": ["x", "y"],
     "rows": [[0.0, 1.0], [1.0, 2.0]]},
    {"path": "a_30deg.csv", "angle": 30.0, "columns": ["x", "y"],
     "rows": [[0.0, 2.0], [1.0, 1.0]]}],
    "settings": {"x_col": "x", "y_col": "y", "x_val": "0.5"}}
PD = {"datasets": {"d1": {"columns": ["x", "y"],
                          "data": [[float(i), float(i) % 3] for i in range(10)]}},
      "axis_selections": {"x_column": "x", "y_column": ["y"]},
      "axis_ranges": {"x_min": 1.0, "x_max": 8.0},
      "plot_settings": {"plot_type": "Line"}}

at = AppTest.from_file(APP, default_timeout=120)
at.query_params["session"] = ";".join([
    write("ang.json", ANG), write("pd1.json", PD), write("pd2.json", PD)])
at.run()
assert not at.exception, [e.value for e in at.exception]

P1, P2, P3 = pid_for("ang.json"), pid_for("pd1.json"), pid_for("pd2.json")

# (accessor, key suffix, new value) widgets to change on panel 2
PD_CHANGES = [
    ("selectbox", "ptype", "Scatter"),
    ("number_input", "xdiv", 3.0),
    ("number_input", "ydiv", 4.0),
    ("radio", "cmode", "Gradient"),
    ("checkbox", "xlog", True),
    ("checkbox", "ylog", True),
    ("number_input", "xmin", 2.0),
    ("number_input", "xmax", 7.0),
    ("number_input", "ymin", 0.5),
    ("number_input", "ymax", 2.5),
    ("checkbox", "grid", False),
    ("slider", "galpha", 0.7),
    ("selectbox", "gls", "--"),
    ("number_input", "xmaj", 2.0),
    ("number_input", "ymaj", 3.0),
    ("number_input", "tsize", 20),
    ("number_input", "legsize", 14),
    ("checkbox", "legshow", False),
    ("text_input", "title", "zzz"),
    ("text_input", "xlabel", "xx"),
]
PD_WATCH = [s for _, s, _ in PD_CHANGES] + ["names", "xcol", "ycols"]

ANG_CHANGES = [
    ("selectbox", "mk", "s"),
    ("selectbox", "tz", "E (0 deg, standard)"),
    ("selectbox", "dir", "Clockwise"),
    ("selectbox", "adisp", "-180 to 180"),
    ("checkbox", "conn", False),
    ("checkbox", "close", True),
    ("checkbox", "roff", True),
    ("text_input", "xval", "0.25"),
    ("text_input", "title", "ang-zzz"),
]
ANG_WATCH = [s for _, s, _ in ANG_CHANGES] + ["xcol", "ycol", "slice"]


def snap(pid, sufs):
    out = {}
    for s in sufs:
        for kind in ("selectbox", "number_input", "checkbox", "text_input",
                     "slider", "radio", "multiselect"):
            try:
                out[s] = getattr(at, kind)(key=f"{pid}-{s}").value
                break
            except Exception:
                continue
    return out


def _same(a, b):
    import math
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or a == b
    return a == b


def sweep(src_pid, changes, watch, *other_pids):
    base = {p: snap(p, watch) for p in other_pids}
    for kind, suf, val in changes:
        w = getattr(at, kind)(key=f"{src_pid}-{suf}")
        w.set_value(val).run()
        assert not at.exception, (suf, [e.value for e in at.exception])
        for p in other_pids:
            now = snap(p, watch)
            for s in base[p]:
                assert _same(now[s], base[p][s]), \
                    f"CROSS-TALK: changed {src_pid}-{suf} -> {p}-{s}: " \
                    f"{base[p][s]!r} -> {now[s]!r}"
    print(f"sweep {src_pid}: no cross-talk "
          f"({len(changes)} changes x {len(other_pids)} panels watched)")


sweep(P2, PD_CHANGES, PD_WATCH, P3)
sweep(P1, ANG_CHANGES, ANG_WATCH, P2, P3)
print("cross-talk probe OK")
