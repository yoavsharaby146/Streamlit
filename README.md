# Session Plotter (Streamlit)

Viewer for session JSON files written by the Tk apps `Plot data.py` and
`angular polar.py` (kept in the [QLMG research](../QLMG%20research/Plotter) repo).
Self-contained: only needs streamlit, numpy, pandas, matplotlib.

## Run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Upload a session JSON in the sidebar (or open `?session=path/to/session.json`).
Modes: "Angular Polar" for sessions with a `files` key, "Plot Data" for
sessions with a `datasets` key.
