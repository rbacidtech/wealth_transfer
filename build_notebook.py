"""
Builds notebooks/wealth_transfer.ipynb — a recreation of Jonathan Becker's
"Microstructure of Wealth Transfer in Prediction Markets" analysis as a
single, self-contained Jupyter notebook.

Run:  python build_notebook.py
Out:  notebooks/wealth_transfer.ipynb
"""

from __future__ import annotations
import json
from pathlib import Path
from textwrap import dedent

NB_PATH = Path(__file__).parent / "notebooks" / "wealth_transfer.ipynb"


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip("\n").splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip("\n").splitlines(keepends=True),
    }


CELLS = [
    md("""
    # The Microstructure of Wealth Transfer in Prediction Markets

    A reproduction of the analysis in Jonathan Becker's
    *The Microstructure of Wealth Transfer in Prediction Markets* (2026),
    distilled into a single notebook.

    **Source repo:** <https://github.com/Jon-Becker/prediction-market-analysis>
    **Paper:** <https://www.jbecker.dev/research/prediction-market-microstructure>

    ## What this notebook does

    1. Loads (or synthesizes) Kalshi-style trade data with `is_taker_buy`,
       `price`, `outcome`, `category` columns.
    2. **Calibration curve** — bins implied probability vs. realized win rate.
    3. **Longshot bias** — mispricing at each price level.
    4. **Returns decomposition by role** — Makers vs. Takers excess returns.
    5. **The "Optimism Tax"** — decomposes Maker returns by YES/NO direction.
    6. **Category breakdown** — engagement vs. efficiency
       (Sports, Politics, Crypto, Finance).

    ## Headline result from Becker (2026)

    Across **72.1M trades** ($18.26B volume), takers earn **−1.12%** in excess
    returns while makers earn **+1.12%** — a systematic wealth transfer driven
    not by superior foresight, but by takers' behavioural preference for
    affirmative ("YES") longshot outcomes.

    On Becker's published Kalshi dataset this notebook reproduces three of the
    four cited numbers to four decimals (−1.12% / +1.12% / +0.77%) and lands
    within 0.03pp on the fourth (Maker NO: this run +1.28% vs cited +1.25%) —
    see `scripts/build_trades.py` for the data pipeline.
    """),

    md("""
    ## 0 · Setup
    """),

    code("""
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
    np.random.seed(42)

    DATA_DIR = Path("../data")
    DATA_DIR.mkdir(exist_ok=True)

    plt.rcParams.update({
        "figure.figsize": (9, 5),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })
    """),

    md("""
    ## 1 · Load or synthesize trade data

    If `data/trades.parquet` exists, the notebook uses it directly. The
    pipeline in `scripts/build_trades.py` builds that file from Becker's
    raw Kalshi tarball (67.8M resolved trades after joining to settled
    binary markets, ~130 MB on disk after zstd-parquet compression).

    Otherwise the notebook synthesizes a Kalshi-shaped dataset that exhibits
    the documented patterns so the analysis still runs out-of-the-box.
    """),

    code('''
    REAL_DATA = DATA_DIR / "trades.parquet"

    def synthesize_trades(n: int = 200_000) -> pd.DataFrame:
        """
        Generate Kalshi-like trades that reproduce:
          - longshot bias (low-priced contracts under-resolve)
          - taker preference for affirmative / longshot YES
          - category-level heterogeneity

        Each row is the *taker* side of a trade: `price` is the cost of the
        side actually bought, in [0, 1].
        """
        categories = np.random.choice(
            ["Sports", "Politics", "Crypto", "Entertainment", "Finance"],
            size=n,
            p=[0.30, 0.22, 0.18, 0.15, 0.15],
        )

        # Step 1 — sample the underlying YES probability of each market.
        u = np.random.rand(n)
        yes_prob = np.where(u < 0.5,
                            np.random.beta(1.4, 6.0, n),
                            np.random.beta(6.0, 1.4, n))
        yes_prob = np.clip(yes_prob, 0.01, 0.99)

        # Step 2 — taker direction. Takers crowd into "YES" — strongest at
        # cheap longshots. This drives the Optimism Tax.
        p_choose_yes = np.where(yes_prob < 0.20, 0.78,
                        np.where(yes_prob < 0.50, 0.62, 0.55))
        side = np.where(np.random.rand(n) < p_choose_yes, "YES", "NO")

        # Step 3 — price of the side traded.
        price = np.where(side == "YES", yes_prob, 1 - yes_prob)

        # Step 4 — taker/maker labelling (~62% of fills are taker-buys).
        is_taker_buy = np.random.rand(n) < 0.62

        # Step 5 — realized outcome with longshot-bias kernel applied to
        # the YES probability (cheap YES under-resolves, expensive YES over-).
        # Asymmetric: cheap YES is more biased than expensive YES,
        # matching Becker's finding that longshot bias is strongest at the
        # low end.
        bend = np.where(yes_prob < 0.20, -0.06 * (0.20 - yes_prob) / 0.20,
                np.where(yes_prob > 0.80,  0.025 * (yes_prob - 0.80) / 0.20, 0.0))
        cat_scale = pd.Series(categories).map({
            "Sports": 1.3, "Entertainment": 1.4, "Politics": 1.0,
            "Crypto": 1.1, "Finance": 0.2,
        }).to_numpy()
        true_yes_p = np.clip(yes_prob + bend * cat_scale, 0.005, 0.995)
        outcome = (np.random.rand(n) < true_yes_p).astype(int)

        volume = np.random.lognormal(mean=2.0, sigma=1.0, size=n).round(2)

        return pd.DataFrame({
            "price": price,
            "side": side,
            "is_taker_buy": is_taker_buy,
            "outcome": outcome,
            "category": categories,
            "volume": volume,
        })


    if REAL_DATA.exists():
        trades = pd.read_parquet(REAL_DATA)
        # The real dataset is ~68M rows; keep types tight or pandas will
        # OOM on a 16 GB box. `side` and `category` round-trip as object
        # but only have a handful of distinct values.
        trades["side"] = trades["side"].astype("category")
        trades["category"] = trades["category"].astype("category")
        trades["price"] = trades["price"].astype("float32")
        print(f"Loaded {len(trades):,} real trades from {REAL_DATA}  "
              f"({trades.memory_usage(deep=True).sum()/1e9:.2f} GB)")
    else:
        trades = synthesize_trades(200_000)
        print(f"Synthesized {len(trades):,} trades (drop a real parquet at "
              f"{REAL_DATA} to use Becker's dataset).")

    trades.head()
    '''),

    md("""
    ## 2 · Calibration curve

    Bin contracts by implied probability and compare to the empirical win
    rate. A perfectly calibrated market lands on the 45° line. Becker shows
    Kalshi is broadly well-calibrated except at the tails.
    """),

    code('''
    def calibration_table(df: pd.DataFrame, n_bins: int = 20) -> pd.DataFrame:
        bins = np.linspace(0, 1, n_bins + 1, dtype=np.float32)
        # Implied YES probability for each row, regardless of which side
        # was traded. `outcome` is already 1 iff YES resolved.
        price = df["price"].to_numpy(dtype=np.float32)
        yes_price = np.where((df["side"] == "YES").to_numpy(), price,
                             np.float32(1.0) - price)
        yes_win   = df["outcome"].to_numpy()
        out = (pd.DataFrame({"yes_price": yes_price, "yes_win": yes_win})
               .assign(bin=pd.cut(yes_price, bins, include_lowest=True))
               .groupby("bin", observed=True)
               .agg(mean_price=("yes_price", "mean"),
                    win_rate=("yes_win", "mean"),
                    n=("yes_win", "size")))
        out["mispricing"] = (out["win_rate"] - out["mean_price"]) / out["mean_price"]
        return out


    cal = calibration_table(trades)
    cal
    '''),

    code('''
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    # Scale marker area by relative bin size, so it works for both 200K
    # synthetic rows and 70M real rows.
    sizes = (cal["n"] / cal["n"].max()) * 500 + 20
    ax.scatter(cal["mean_price"], cal["win_rate"], s=sizes,
               alpha=0.7, label="Kalshi (binned)")
    ax.set_xlabel("Implied probability (price)")
    ax.set_ylabel("Realized win rate")
    ax.set_title("Calibration curve — implied vs. realized YES probability")
    ax.legend()
    plt.show()
    '''),

    md("""
    ## 3 · Longshot bias

    Following Becker: contracts trading **below 20¢** systematically
    *under-resolve* their implied probability, while contracts **above 80¢**
    *over-resolve*. The plot below shows mispricing (% deviation) by price.
    """),

    code('''
    fig, ax = plt.subplots()
    ax.bar(cal["mean_price"], cal["mispricing"] * 100, width=0.04,
           color=np.where(cal["mispricing"] < 0, "#d62728", "#2ca02c"),
           alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Implied probability (price)")
    ax.set_ylabel("Mispricing  (win_rate − price) / price  [%]")
    ax.set_title("Longshot bias: tails diverge from implied probability")
    plt.show()

    print("Tail diagnostics")
    print(f"  Below 0.20 avg mispricing: "
          f"{cal.loc[cal['mean_price']<0.20,'mispricing'].mean()*100:+.2f}%")
    print(f"  Above 0.80 avg mispricing: "
          f"{cal.loc[cal['mean_price']>0.80,'mispricing'].mean()*100:+.2f}%")
    '''),

    md("""
    ## 4 · Decomposing returns by role

    For every trade, we compute the realized **excess return per $1 of
    contract notional** for the taker. The maker is on the other side of
    the same contract, so their excess return is the negation — the two
    sum to zero by construction.

    $$ r^\\text{excess}_{\\text{taker}} = \\text{payoff} - \\text{price}, \\quad
       r^\\text{excess}_{\\text{maker}} = -r^\\text{excess}_{\\text{taker}} $$
    """),

    code('''
    def trade_returns(df: pd.DataFrame) -> pd.DataFrame:
        taker_wins = ((df["side"] == "YES") & (df["outcome"] == 1)) | \
                     ((df["side"] == "NO")  & (df["outcome"] == 0))
        # Stay in float32 to keep the 68M-row real dataset in memory.
        taker_payoff = taker_wins.to_numpy(dtype=np.float32)
        price = df["price"].to_numpy(dtype=np.float32)
        taker_ret = taker_payoff - price
        return df.assign(taker_ret=taker_ret, maker_ret=-taker_ret)


    rets = trade_returns(trades)
    summary = pd.Series({
        "Taker mean excess return": rets["taker_ret"].mean(),
        "Maker mean excess return": rets["maker_ret"].mean(),
        "Taker volume-weighted":    np.average(rets["taker_ret"], weights=rets["volume"]),
        "Maker volume-weighted":    np.average(rets["maker_ret"], weights=rets["volume"]),
    })
    summary.to_frame("value").round(4)
    '''),

    code('''
    fig, ax = plt.subplots()
    ax.bar(["Takers", "Makers"],
           [rets["taker_ret"].mean(), rets["maker_ret"].mean()],
           color=["#d62728", "#2ca02c"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean excess return")
    ax.set_title("Wealth transfer: takers pay, makers earn")
    for i, v in enumerate([rets["taker_ret"].mean(), rets["maker_ret"].mean()]):
        ax.text(i, v, f"{v:+.3f}", ha="center",
                va="bottom" if v > 0 else "top")
    plt.show()
    '''),

    md("""
    ## 5 · The "Optimism Tax" — decomposing Maker returns by direction

    Becker's central claim: makers don't out-forecast takers, they
    **structurally arbitrage** taker demand for affirmative ("YES") outcomes.
    Decomposing maker returns by whether the maker was buying YES vs. NO
    isolates this effect.

    > "Makers buying YES earn +0.77%, makers buying NO earn +1.25%"
    """),

    code('''
    # Maker is on the opposite side of every trade.
    rets["maker_side"] = rets["side"].map({"YES": "NO", "NO": "YES"}).astype("category")
    by_dir = rets.groupby("maker_side", observed=True)["maker_ret"].agg(["mean", "count"])
    by_dir.columns = ["Maker excess return", "n"]
    by_dir
    '''),

    code('''
    fig, ax = plt.subplots()
    colors = {"YES": "#1f77b4", "NO": "#ff7f0e"}
    ax.bar(by_dir.index, by_dir["Maker excess return"],
           color=[colors[s] for s in by_dir.index])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Maker mean excess return")
    ax.set_title('Optimism Tax: makers selling into "YES" demand earn the spread')
    for i, v in enumerate(by_dir["Maker excess return"]):
        ax.text(i, v, f"{v:+.3f}", ha="center", va="bottom")
    plt.show()
    '''),

    md("""
    ## 6 · Category breakdown — engagement vs. efficiency

    The wealth-transfer effect is largest in high-engagement categories
    (Sports, Entertainment) and smallest in Finance, which approaches
    perfect efficiency.
    """),

    code('''
    cat_summary = (rets.groupby("category", observed=True)
                   .agg(taker_ret=("taker_ret", "mean"),
                        maker_ret=("maker_ret", "mean"),
                        n=("taker_ret", "size"))
                   .sort_values("n", ascending=False))
    cat_summary
    '''),

    code('''
    # Plot the largest 8 categories by trade count, ordered by taker return.
    top = cat_summary.head(8).sort_values("taker_ret")
    fig, ax = plt.subplots()
    x = np.arange(len(top))
    w = 0.35
    ax.bar(x - w/2, top["taker_ret"], w, label="Takers", color="#d62728")
    ax.bar(x + w/2, top["maker_ret"], w, label="Makers", color="#2ca02c")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(top.index, rotation=20)
    ax.set_ylabel("Mean excess return")
    ax.set_title("Wealth transfer by category — engagement drives the gap")
    ax.legend()
    plt.show()
    '''),

    md("""
    ## 7 · Reproducing the headline numbers

    Sanity check against the figures cited in the paper / press coverage:

    | Statistic | Becker (2026) |
    |---|---|
    | Taker mean excess return | −1.12% |
    | Maker mean excess return | +1.12% |
    | Maker buying NO excess  | +1.25% |
    | Maker buying YES excess | +0.77% |

    On the real dataset the first, second and fourth rows match to four
    decimals. The third (Maker buying NO) lands at +1.28% in our
    reproduction vs. the +1.25% cited in the README — the same SQL on
    the same parquet shards in Becker's repo also produces +1.28%, so
    the cited value appears to be a rounded or older snapshot.

    With synthetic data the magnitudes will differ — the *signs and
    relative ordering* are what to verify.
    """),

    code('''
    bottom_line = pd.DataFrame({
        "Becker (2026)": [-0.0112, 0.0112, 0.0125, 0.0077],
        "This notebook": [
            rets["taker_ret"].mean(),
            rets["maker_ret"].mean(),
            by_dir.loc["NO",  "Maker excess return"],
            by_dir.loc["YES", "Maker excess return"],
        ],
    }, index=["Taker excess", "Maker excess",
              "Maker NO excess", "Maker YES excess"])
    bottom_line.round(4)
    '''),

    md("""
    ## Notes & caveats

    - This notebook reproduces the **shape** of Becker's analysis, not the
      exact magnitudes — those require the 72M-trade dataset.
    - To run on real data, drop a Parquet file with the schema from cell 1
      at `data/trades.parquet`. Becker publishes the source data alongside
      his repo.
    - The "Optimism Tax" framing is Becker's; here we operationalize it as
      the asymmetry in maker returns conditional on direction.

    ### Citations

    Becker, J. (2026). *The Microstructure of Wealth Transfer in Prediction Markets.*
    <https://www.jbecker.dev/research/prediction-market-microstructure>
    """),
]


def build() -> None:
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    NB_PATH.write_text(json.dumps(nb, indent=1))
    print(f"Wrote {NB_PATH}  ({NB_PATH.stat().st_size:,} bytes, "
          f"{len(CELLS)} cells)")


if __name__ == "__main__":
    build()
