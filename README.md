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
- Every panel is **fully independent**: all settings (columns, plot type,
  dividers, log/grid, axis ranges) live in the panel's own expanders and
  are remembered per browser session. Each panel's axis ranges follow its
  current X/Y/Z column.
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
  multi-column mode); dragging it moves only its own panel.

## Performance & polish

- **Demo sessions**: a sidebar button loads two synthetic example sessions
  (Angular Polar + Plot Data) so the app can be tried without a session
  file — click again to remove them.
- **Figure theme** (sidebar): Dark redraws text/ticks light and makes the
  figure background transparent so plots match a dark page; Auto follows
  `theme.base` when set in `.streamlit/config.toml`.
- Figure renders are **cached**: cosmetic edits, other panels' settings and
  scrubbing back to an already-seen X value reuse earlier renders instead
  of redrawing (one render also serves the Download PNG button).
- The polar **X-slice slider reruns only its own panel** (Streamlit
  fragment), so scrubbing stays instant even with several panels loaded.
- Each panel's settings are remembered per browser session under an id
  derived from the session **name** (not its position), so swapping or
  reordering files never mixes up remembered values; a panel's
  **↺ Reset to file values** button forgets them and reloads the ranges,
  labels and styles saved in the session JSON.

## Self-check

```bash
python streamlit_app.py --self-check
```

