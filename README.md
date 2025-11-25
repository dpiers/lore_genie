# Lorcana Odds Tool

Simple FastAPI service plus a static HTML front-end to explore pack odds for Disney Lorcana sets.

## Quick start

Using `uv` (recommended):

```bash
# from the repo root
UV_CACHE_DIR=.uv-cache uv run -- uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```

Then visit `http://127.0.0.1:8000/` to load the visualizer (the API also serves the static HTML under `/web`).

For live-reload while editing HTML/CSS/JS in `web/`, include reload patterns:

```bash
UV_CACHE_DIR=.uv-cache uv run -- uvicorn api:app \
  --host 127.0.0.1 --port 8000 --reload \
  --reload-include 'web/*.html' --reload-include 'web/*.js' --reload-include 'web/*.css'
```
The server restarts on changes; refresh your browser to see them.

Using plain pip:

1. (Optional) Create and activate a virtualenv for Python 3.12.
2. Install dependencies: `pip install -r requirements.txt` (or `pip install .` to use `pyproject.toml`).
3. Run the API: `python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000`
4. Open `web/mc_visualizer.html` in your browser and point it at the API (defaults to `http://localhost:8000`).

The API supports:
- `GET /sets` for set metadata and rarities
- `POST /mc-batch` to run a Monte Carlo histogram for a rarity and pack count
- `POST /inspect-rarity` for quick analytic probabilities
