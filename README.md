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
  (The picker accepts any file — handy on phones, where cloud providers
  like Dropbox don't flag files as .json; contents are validated on load.)
- Or pass files in the URL (repeatable, or `;`-separated on Windows):
  `?session=path/a.json;path/b.json`.
- Add `&cols=3` (or `&cols=2`) to force the panels-per-row layout when
  sharing a link.

## Comparing sessions

- **Panels per row** (1–3): lay the loaded sessions out side by side.
- **Sync settings across panels**: one shared control block (X/Y columns,
  x slice values, slice mode, theta zero/direction/angle display, plot
  type, dividers, log/grid) applied to every panel — e.g. compare the
  same x-slice across several angle scans. Turn it off for fully
  independent per-panel settings.
- Each panel has its own **Download PNG** button; a bad session only
  disables its own panel.
- **Panel layout:** each panel shows its **plot on top** (X-slice slider
  and Download PNG right under it); every setting lives in collapsible
  expanders *below* the plot, so panels line up side by side even when
  the panel types differ.
- **Plot Data panels** support Line / Scatter / **Color Map** (Z column in
  color with a colorbar, gridded like `Plot data.py`; regular sweep grids
  are reshaped exactly, scattered data is interpolated).
- Polar panels show an **X-slice slider right under the plot** (also in
  multi-column mode). With sync on, dragging it updates the shared X —
  every polar panel scrubs together; with sync off it moves only its own
  panel.

## Self-check

```bash
python streamlit_app.py --self-check
```

