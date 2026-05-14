# wealth_transfer

A standalone Jupyter notebook that recreates Jonathan Becker's analysis from
*The Microstructure of Wealth Transfer in Prediction Markets* (2026):
calibration curve, longshot bias, Maker/Taker returns decomposition, and the
"Optimism Tax."

- Source repo: <https://github.com/Jon-Becker/prediction-market-analysis>
- Paper: <https://www.jbecker.dev/research/prediction-market-microstructure>

## Quick start (synthetic data)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python build_notebook.py          # writes notebooks/wealth_transfer.ipynb
jupyter lab notebooks/wealth_transfer.ipynb
```

The notebook synthesizes a Kalshi-shaped dataset by default, so it runs
out of the box and demonstrates the *shape* of Becker's analysis.

## Running against Becker's real Kalshi dataset

1. Place Becker's compressed data tarball at the repo root as
   `data.tar.zst` (the upstream repo distributes `data.tar` which can be
   re-compressed with `zstd data.tar`).
2. Extract just the Kalshi subset:
   ```bash
   zstd -dc data.tar.zst | tar -xf - --wildcards 'data/kalshi/*' --exclude='*/._*'
   ```
3. Build `data/trades.parquet` (joins markets ↔ trades, applies Becker's
   category map, writes a single ~130 MB zstd-parquet):
   ```bash
   python scripts/build_trades.py
   ```
4. Re-run `python build_notebook.py` (or just open the notebook). The
   load cell now finds `data/trades.parquet` and uses the real 67.8M
   resolved trades instead of synthetic data.

The output schema produced by `scripts/build_trades.py` (also the
synthetic schema) is:

| column | type | meaning |
|---|---|---|
| `price` | float32 | cost of the side the taker bought, in [0, 1] |
| `side` | category | "YES" or "NO" — which side the taker bought |
| `is_taker_buy` | bool | always True for Kalshi (the taker is always the buyer of their stated side) |
| `outcome` | int8 | 1 if the market resolved YES, 0 if NO |
| `category` | category | Becker's group label (Sports, Crypto, Politics, Finance, …) |
| `volume` | int32 | trade size in contracts |

## Layout

```
.
├── build_notebook.py            # programmatically builds the .ipynb
├── scripts/
│   ├── build_trades.py          # raw kalshi parquets → data/trades.parquet
│   └── _becker_categories.py    # vendored from Becker's repo (MIT)
├── notebooks/
│   └── wealth_transfer.ipynb    # generated
├── data/                        # trades.parquet lands here
├── requirements.txt
└── README.md
```
