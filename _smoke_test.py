"""Headless smoke test for streamlit_app.py using Streamlit's AppTest."""
import hashlib

from streamlit.testing.v1 import AppTest


def pid_for(name):
    """Same content-addressed pid the app assigns (see _panel_pid)."""
    return "p-" + hashlib.md5(name.encode("utf-8")).hexdigest()[:6]


PID_ANG, PID_PD = pid_for("demo_angular.json"), pid_for("demo_plotdata.json")

at = AppTest.from_file(r"c:\Users\Yoav\VScode\streamlit-plotter\streamlit_app.py",
                       default_timeout=120)
at.run()
assert not at.exception, [e.value for e in at.exception]

# 1) load the demo sessions
btn = [b for b in at.sidebar.button if "demo" in b.label.lower()][0]
assert btn.label == "Load demo sessions"
btn.click()
at.run()
assert not at.exception, [e.value for e in at.exception]
assert at.session_state["demo-on"] is True

# 2) figure theme: Dark / Light / Auto all render without errors
ts = [s for s in at.sidebar.selectbox if s.label == "Figure theme"][0]
ts.set_value("Dark").run()
assert not at.exception, [e.value for e in at.exception]
ts.set_value("Light").run()
assert not at.exception, [e.value for e in at.exception]
ts.set_value("Auto").run()
assert not at.exception, [e.value for e in at.exception]

# 3) polar X-slice slider (demo angular panel): drag to a new value
sl = at.slider(key=f"{PID_ANG}-xslider")
sl.set_value(6.0).run()
assert not at.exception, [e.value for e in at.exception]

# 4) switch the Plot Data panel to Color Map (panels are independent now)
pt = at.selectbox(key=f"{PID_PD}-ptype")
pt.set_value("Color Map").run()
assert not at.exception, [e.value for e in at.exception]

# 4.5) per-column sticky ranges follow the X column (the reconciliation
# block swaps the remembered label + range in whenever the column changes)
xc = at.selectbox(key=f"{PID_PD}-xcol")
xc.set_value("background").run()
assert not at.exception, [e.value for e in at.exception]
xc.set_value("x").run()
assert not at.exception, [e.value for e in at.exception]
# demo file's saved x range (1..9) came back with the column
assert at.number_input(key=f"{PID_PD}-xmin").value == 1.0
assert at.number_input(key=f"{PID_PD}-xmax").value == 9.0

# 5) reset the Plot Data panel to file values (clears remembered state)
rb = at.button(key=f"{PID_PD}-resetbtn")
rb.click()
at.run()
assert not at.exception, [e.value for e in at.exception]
# file default was a Line plot, so the reset restored it
assert at.selectbox(key=f"{PID_PD}-ptype").value == "Line"

# 6) back to clean state: remove demos
btn = [b for b in at.sidebar.button if "demo" in b.label.lower()][0]
assert btn.label == "Remove demo sessions"
btn.click()
at.run()
assert not at.exception, [e.value for e in at.exception]
assert at.session_state["demo-on"] is False

# 6) duplicate-basename URL sessions: distinct panel ids, and resetting
# one panel never touches its same-named twin's settings
import json as _json
import os as _os
import tempfile as _tf

_ses = {"files": [{"path": "a_0deg.csv", "angle": 0.0,
                   "columns": ["x", "y"],
                   "rows": [[0.0, 1.0], [1.0, 2.0]]}],
        "settings": {"x_col": "x", "y_col": "y", "x_val": "0.5"}}
_paths = []
for _ in (1, 2):
    _d = _tf.mkdtemp()
    _p = _os.path.join(_d, "dup.json")
    with open(_p, "w") as _fh:
        _json.dump(_ses, _fh)
    _paths.append(_p)
at2 = AppTest.from_file(r"c:\Users\Yoav\VScode\streamlit-plotter\streamlit_app.py",
                        default_timeout=120)
at2.query_params["session"] = ";".join(_paths)
at2.run()
assert not at2.exception, [e.value for e in at2.exception]
D1, D2 = pid_for("dup.json"), pid_for("dup.json (#2)")
assert at2.slider(key=f"{D1}-xslider").value == 0.5  # both loaded, file x_val
at2.slider(key=f"{D2}-xslider").set_value(0.75).run()
assert not at2.exception, [e.value for e in at2.exception]
at2.button(key=f"{D1}-resetbtn").click()  # reset twin #1...
at2.run()
assert not at2.exception, [e.value for e in at2.exception]
# ...twin #2 keeps its scrubbed value, twin #1 is back at the file value
assert at2.slider(key=f"{D2}-xslider").value == 0.75
assert at2.slider(key=f"{D1}-xslider").value == 0.5

# 7) three DISTINCT-name panels (angular + two Plot Data): resetting any
# one panel never changes the other panels' settings or manual ranges
def _write(name, obj):
    p = _os.path.join(_tf.mkdtemp(), name)
    with open(p, "w") as fh:
        _json.dump(obj, fh)
    return p


_ang3 = {"files": [
    {"path": "a_0deg.csv", "angle": 0.0, "columns": ["x", "y"],
     "rows": [[0.0, 1.0], [1.0, 2.0]]},
    {"path": "a_30deg.csv", "angle": 30.0, "columns": ["x", "y"],
     "rows": [[0.0, 2.0], [1.0, 1.0]]}],
    "settings": {"x_col": "x", "y_col": "y", "x_val": "0.5"}}
_pd = {"datasets": {"d1": {"columns": ["x", "y"],
                           "data": [[float(i), float(i) % 3] for i in range(10)]}},
       "axis_selections": {"x_column": "x", "y_column": ["y"]},
       "axis_ranges": {"x_min": 1.0, "x_max": 8.0},
       "plot_settings": {"plot_type": "Line"}}
at3 = AppTest.from_file(r"c:\Users\Yoav\VScode\streamlit-plotter\streamlit_app.py",
                        default_timeout=120)
at3.query_params["session"] = ";".join([
    _write("ang.json", _ang3), _write("pd1.json", _pd),
    _write("pd2.json", _pd)])
at3.run()
assert not at3.exception, [e.value for e in at3.exception]
P1 = pid_for("ang.json")
P2, P3 = pid_for("pd1.json"), pid_for("pd2.json")
# both plot panels loaded the file's saved x range
assert at3.number_input(key=f"{P2}-xmin").value == 1.0
assert at3.number_input(key=f"{P3}-xmin").value == 1.0
# manual override on panel 3
at3.number_input(key=f"{P3}-xmin").set_value(2.5).run()
assert not at3.exception, [e.value for e in at3.exception]
# resetting panel 1 (angular) and panel 2 leaves panel 3 untouched
at3.button(key=f"{P1}-resetbtn").click()
at3.run()
assert not at3.exception, [e.value for e in at3.exception]
assert at3.number_input(key=f"{P3}-xmin").value == 2.5
at3.button(key=f"{P2}-resetbtn").click()
at3.run()
assert not at3.exception, [e.value for e in at3.exception]
assert at3.number_input(key=f"{P3}-xmin").value == 2.5
assert at3.number_input(key=f"{P2}-xmin").value == 1.0  # its own reset worked

print("AppTest smoke OK")
