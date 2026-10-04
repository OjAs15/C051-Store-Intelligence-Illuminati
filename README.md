# Store Expansion Intelligence — V2 Real-Data PoC

Locked case: Croma; existing stores = Prabhadevi + Powai; candidates = BKC + Dadar + Vikhroli; common catchment = 2 km.

The starter is intentionally modular: acquisition → validation → location intelligence → ML demand prediction → five-engine decision flow. Replace the placeholder connectors in `src/acquisition.py` one source at a time. Do not invent missing public data.

## Local run
`python -m venv .venv`
`..\.venv\Scripts\Activate.ps1` (PowerShell)
`pip install -r requirements.txt`
`pytest -q`
`python scripts/run_v2_refresh.py`
`streamlit run app.py`

The ML component is deliberately a synthetic-demo scaffold until a legitimate historical training set exists; do not label it as a Croma-calibrated sales forecast.
