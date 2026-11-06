# Session Plotter (Streamlit)

Viewer for session JSON files written by the Tk apps `Plot data.py` and
`angular polar.py` (kept in the [QLMG research](../QLMG%20research/Plotter) repo).
Self-contained: only needs streamlit, numpy, pandas, matplotlib.

## Run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Loading sessions

- Upload one **or several** session JSONs in the sidebar — each becomes a
  plot panel, and the session type is detected automatically
  (`files` key = Angular Polar, `datasets` key = Plot Data).
- Or pass files in the URL (repeatable, or `;`-separated on Windows):
  `?session=path/a.json;path/b.json`.

## Comparing sessions

- **Panels per row** (1–3): lay the loaded sessions out side by side.
- **Sync settings across panels**: one shared control block (X/Y columns,
  x slice values, slice mode, theta zero/direction/angle display, plot
  type, dividers, log/grid) applied to every panel — e.g. compare the
  same x-slice across several angle scans. Turn it off for fully
  independent per-panel settings.
- Each panel has its own **Download PNG** button; a bad session only
  disables its own panel.

## Self-check

```bash
python streamlit_app.py --self-check
```

