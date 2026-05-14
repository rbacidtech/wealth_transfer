"""
Build data/trades.parquet from Becker's Kalshi data tarball.

Reads:
  data/kalshi/markets/*.parquet
  data/kalshi/trades/*.parquet

Writes:
  data/trades.parquet  (columns: price, side, is_taker_buy, outcome, category, volume)

Uses DuckDB for the join (matches Jon Becker's prediction-market-analysis SQL
verbatim) and Becker's published category mapping
(github.com/Jon-Becker/prediction-market-analysis/.../util/categories.py).

Run:
  .venv/bin/python scripts/build_trades.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
MARKETS_DIR = ROOT / "data" / "kalshi" / "markets"
TRADES_DIR = ROOT / "data" / "kalshi" / "trades"
OUT_PATH = ROOT / "data" / "trades.parquet"

# Import Becker's full SUBCATEGORY_PATTERNS list (vendored at scripts/_becker_categories.py).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _becker_categories import SUBCATEGORY_PATTERNS  # noqa: E402


def build_category_case_sql() -> str:
    """Convert Becker's SUBCATEGORY_PATTERNS (Python substring matching, in order)
    into one big DuckDB CASE expression that maps event_ticker prefix → group.

    The Python logic is: `if pattern in cat_upper` — first match wins.
    The SQL equivalent: nested CASE WHEN with INSTR() to test substring, in the
    same order.
    """
    # Deduplicate by pattern (the list has a few duplicates).
    seen: set[str] = set()
    parts: list[str] = []
    for pattern, group, _cat, _subcat in SUBCATEGORY_PATTERNS:
        if pattern in seen:
            continue
        seen.add(pattern)
        # Escape single quotes (none in the patterns, but be safe).
        p = pattern.replace("'", "''")
        g = group.replace("'", "''")
        parts.append(f"WHEN instr(prefix, '{p}') > 0 THEN '{g}'")
    return "CASE\n        " + "\n        ".join(parts) + "\n        ELSE 'Other'\n      END"


def main() -> None:
    if not MARKETS_DIR.exists() or not any(MARKETS_DIR.glob("*.parquet")):
        sys.exit(f"missing: {MARKETS_DIR}")
    if not TRADES_DIR.exists() or not any(TRADES_DIR.glob("*.parquet")):
        sys.exit(f"missing: {TRADES_DIR}")

    category_case = build_category_case_sql()

    con = duckdb.connect()
    con.execute("PRAGMA threads=6")
    con.execute("PRAGMA memory_limit='10GB'")

    sql = f"""
    COPY (
      WITH resolved_markets AS (
        SELECT ticker,
               event_ticker,
               result,
               regexp_extract(coalesce(event_ticker, ticker), '^([A-Z0-9]+)', 1) AS prefix
        FROM '{MARKETS_DIR}/*.parquet'
        WHERE status = 'finalized'
          AND result IN ('yes', 'no')
      ),
      markets_with_group AS (
        SELECT ticker, result, {category_case} AS category
        FROM resolved_markets
      )
      SELECT
        (CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END) / 100.0 AS price,
        upper(t.taker_side) AS side,
        TRUE AS is_taker_buy,
        (CASE WHEN m.result = 'yes' THEN 1 ELSE 0 END)::TINYINT AS outcome,
        m.category AS category,
        t.count::INTEGER AS volume
      FROM '{TRADES_DIR}/*.parquet' t
      INNER JOIN markets_with_group m ON t.ticker = m.ticker
    ) TO '{OUT_PATH}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """

    print(f"writing {OUT_PATH} ...")
    t0 = time.time()
    con.execute(sql)
    print(f"done in {time.time()-t0:.1f}s")
    sz = OUT_PATH.stat().st_size
    print(f"output: {sz/1e9:.2f} GB")

    # Quick verification — global taker excess should be ~-1.12% (Becker's headline).
    print("\n=== verification ===")
    out = con.execute(f"""
      SELECT
        COUNT(*) AS n,
        AVG((CASE WHEN side = 'YES' AND outcome = 1 THEN 1.0
                   WHEN side = 'NO'  AND outcome = 0 THEN 1.0
                   ELSE 0.0 END) - price) AS taker_excess,
        AVG(-((CASE WHEN side = 'YES' AND outcome = 1 THEN 1.0
                     WHEN side = 'NO'  AND outcome = 0 THEN 1.0
                     ELSE 0.0 END) - price)) AS maker_excess
      FROM '{OUT_PATH}'
    """).df()
    print(out.to_string(index=False))

    print("\nCategory distribution:")
    print(con.execute(f"""
      SELECT category, COUNT(*) AS n, AVG(volume) AS avg_volume
      FROM '{OUT_PATH}'
      GROUP BY category
      ORDER BY n DESC
    """).df().to_string(index=False))


if __name__ == "__main__":
    main()
