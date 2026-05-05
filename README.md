# wealth_transfer

A standalone Jupyter notebook that recreates Jonathan Becker's analysis from
*The Microstructure of Wealth Transfer in Prediction Markets* (2026):
calibration curve, longshot bias, Maker/Taker returns decomposition, and the
"Optimism Tax."

- Source repo: <https://github.com/Jon-Becker/prediction-market-analysis>
- Paper: <https://www.jbecker.dev/research/prediction-market-microstructure>

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python build_notebook.py          # writes notebooks/wealth_transfer.ipynb
jupyter lab notebooks/wealth_transfer.ipynb
```

The notebook synthesizes a Kalshi-shaped dataset by default, so it runs out
of the box. To run against Becker's real dataset, drop a Parquet file at
`data/trades.parquet` with columns:

| column | type | meaning |
|---|---|---|
| `price` | float | trade price in [0, 1], implied YES probability |
| `side` | str | "YES" or "NO" |
| `is_taker_buy` | bool | True if the aggressor was the buyer |
| `outcome` | int | 1 if market resolved YES, 0 otherwise |
| `category` | str | e.g. Sports, Politics, Crypto, Finance |
| `volume` | float | trade size |

## Layout

```
.
├── build_notebook.py         # programmatically builds the .ipynb
├── notebooks/
│   └── wealth_transfer.ipynb # generated
├── data/                     # drop trades.parquet here for real data
├── requirements.txt
└── README.md
```
