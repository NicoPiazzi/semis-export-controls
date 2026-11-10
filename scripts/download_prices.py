"""
Download daily prices for the semiconductor export-controls event study.

Usage (from the repository root):
    python scripts/download_prices.py
    python scripts/download_prices.py --start 2018-01-01 --end 2026-11-10

Output (in data/raw/):
    prices_long.csv     one row per (date, ticker): open, high, low, close, adj_close, volume
    tickers_meta.csv    role in the supply chain, exchange, currency, benchmark index
    download_log.json   library versions, rows per ticker, first/last date, quality flags

Design choices (see README):
    * Local listings are the main sample: each stock is matched with the broad
      index of its own market, so that both returns are measured on the same
      trading calendar and in the same currency.
    * US-listed ADRs (ASML, TSM) and SMIC's Shanghai line are kept for
      robustness checks.
    * Prices are saved raw (close and adjusted close). Returns are computed later,
      in the analysis step, never here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    sys.exit("yfinance is missing: run  pip install -r requirements.txt")


# --------------------------------------------------------------------------
# Universe
# --------------------------------------------------------------------------
# role: equipment (tools), manufacturer (fabs: foundry and memory), designer (chip design),
#       index (benchmark)
# benchmark: market index used in the market model for that stock
TICKERS = [
    # ---- Main sample: local listings ----------------------------------------
    dict(ticker="ASML.AS",   name="ASML Holding",          role="equipment", sample="main",
         exchange="Euronext Amsterdam", country="NL", currency="EUR", benchmark="^STOXX"),
    dict(ticker="AMAT",      name="Applied Materials",     role="equipment", sample="main",
         exchange="Nasdaq", country="US", currency="USD", benchmark="^GSPC"),
    dict(ticker="LRCX",      name="Lam Research",          role="equipment", sample="main",
         exchange="Nasdaq", country="US", currency="USD", benchmark="^GSPC"),
    dict(ticker="KLAC",      name="KLA",                   role="equipment", sample="main",
         exchange="Nasdaq", country="US", currency="USD", benchmark="^GSPC"),
    dict(ticker="8035.T",    name="Tokyo Electron",        role="equipment", sample="main",
         exchange="Tokyo", country="JP", currency="JPY", benchmark="^N225"),
    dict(ticker="2330.TW",   name="TSMC",                  role="manufacturer",   sample="main",
         exchange="Taiwan", country="TW", currency="TWD", benchmark="^TWII"),
    dict(ticker="005930.KS", name="Samsung Electronics",   role="manufacturer",   sample="main",
         exchange="Korea", country="KR", currency="KRW", benchmark="^KS11"),
    dict(ticker="000660.KS", name="SK Hynix",              role="manufacturer", sample="main",
         exchange="Korea", country="KR", currency="KRW", benchmark="^KS11"),
    dict(ticker="0981.HK",   name="SMIC",                  role="manufacturer",   sample="main",
         exchange="Hong Kong", country="CN", currency="HKD", benchmark="^HSI"),
    dict(ticker="NVDA",      name="NVIDIA",                role="designer",  sample="main",
         exchange="Nasdaq", country="US", currency="USD", benchmark="^GSPC"),
    dict(ticker="AMD",       name="AMD",                   role="designer",  sample="main",
         exchange="Nasdaq", country="US", currency="USD", benchmark="^GSPC"),
    # ---- Robustness: other listings -----------------------------------------
    dict(ticker="ASML",      name="ASML Holding (NY shares)", role="equipment", sample="robustness",
         exchange="Nasdaq", country="NL", currency="USD", benchmark="^GSPC"),
    dict(ticker="TSM",       name="TSMC (ADR)",            role="manufacturer",   sample="robustness",
         exchange="NYSE", country="TW", currency="USD", benchmark="^GSPC"),
    dict(ticker="688981.SS", name="SMIC (Shanghai STAR)",  role="manufacturer",   sample="robustness",
         exchange="Shanghai", country="CN", currency="CNY", benchmark="000001.SS"),
    # ---- Benchmarks ----------------------------------------------------------
    dict(ticker="^GSPC",     name="S&P 500",               role="index", sample="benchmark",
         exchange="US", country="US", currency="USD", benchmark=""),
    dict(ticker="^STOXX",    name="STOXX Europe 600",      role="index", sample="benchmark",
         exchange="Europe", country="EU", currency="EUR", benchmark=""),
    dict(ticker="^AEX",      name="AEX (Amsterdam)",       role="index", sample="benchmark",
         exchange="Euronext Amsterdam", country="NL", currency="EUR", benchmark=""),
    dict(ticker="^N225",     name="Nikkei 225",            role="index", sample="benchmark",
         exchange="Tokyo", country="JP", currency="JPY", benchmark=""),
    dict(ticker="^TWII",     name="TAIEX",                 role="index", sample="benchmark",
         exchange="Taiwan", country="TW", currency="TWD", benchmark=""),
    dict(ticker="^KS11",     name="KOSPI",                 role="index", sample="benchmark",
         exchange="Korea", country="KR", currency="KRW", benchmark=""),
    dict(ticker="^HSI",      name="Hang Seng",             role="index", sample="benchmark",
         exchange="Hong Kong", country="HK", currency="HKD", benchmark=""),
    dict(ticker="000001.SS", name="SSE Composite",         role="index", sample="benchmark",
         exchange="Shanghai", country="CN", currency="CNY", benchmark=""),
    dict(ticker="ACWI",      name="iShares MSCI ACWI ETF", role="index", sample="benchmark",
         exchange="Nasdaq", country="US", currency="USD", benchmark=""),
    dict(ticker="^SOX",      name="PHLX Semiconductor",    role="index", sample="benchmark",
         exchange="Nasdaq", country="US", currency="USD", benchmark=""),
]

PRICE_COLS = ["open", "high", "low", "close", "adj_close", "volume"]


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------
def download_one(ticker: str, start: str, end: str, retries: int = 3) -> pd.DataFrame:
    """Download one ticker and return a tidy DataFrame (date + PRICE_COLS)."""
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            raw = yf.download(
                ticker,
                start=start,
                end=end,            # yfinance treats `end` as exclusive
                interval="1d",
                auto_adjust=False,  # keep both Close and Adj Close
                actions=False,
                progress=False,
                threads=False,
            )
            if raw is None or raw.empty:
                raise ValueError("empty response")
            return tidy(raw, ticker)
        except Exception as exc:  # network errors, rate limits, empty data
            last_error = exc
            wait = 2 ** attempt
            print(f"  {ticker}: attempt {attempt} failed ({exc}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"{ticker}: download failed after {retries} attempts: {last_error}")


def tidy(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Flatten yfinance output (single or MultiIndex columns) into long format."""
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        # recent yfinance versions return (field, ticker) even for one ticker
        level = 0 if "Close" in df.columns.get_level_values(0) else 1
        df.columns = df.columns.get_level_values(level)
    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low", "Close": "close",
        "Adj Close": "adj_close", "Volume": "volume",
    })
    if "adj_close" not in df.columns:
        # should not happen with auto_adjust=False, but keep the pipeline alive
        print(f"  WARNING {ticker}: no Adj Close column, using Close")
        df["adj_close"] = df["close"]
    df = df[[c for c in PRICE_COLS if c in df.columns]]
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["close"])
    df.insert(0, "ticker", ticker)
    return df.reset_index()


# --------------------------------------------------------------------------
# Quality checks
# --------------------------------------------------------------------------
def quality_report(df: pd.DataFrame, jump_threshold: float = 0.25, gap_days: int = 7) -> dict:
    """Simple checks to read before any analysis. Flags, never fixes."""
    s = df.set_index("date")["adj_close"]
    ret = s.pct_change()
    gaps = s.index.to_series().diff().dt.days
    return {
        "rows": int(len(df)),
        "first_date": str(s.index.min().date()) if len(s) else None,
        "last_date": str(s.index.max().date()) if len(s) else None,
        "non_positive_prices": int((s <= 0).sum()),
        "missing_adj_close": int(s.isna().sum()),
        "zero_volume_days": int((df["volume"] == 0).sum()) if "volume" in df else None,
        "large_moves": [
            {"date": str(d.date()), "return": round(float(r), 4)}
            for d, r in ret[ret.abs() > jump_threshold].items()
        ],
        "calendar_gaps_over_7_days": [
            {"date": str(d.date()), "days": int(g)} for d, g in gaps[gaps > gap_days].items()
        ],
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", default="2018-01-01",
                        help="first date (default 2018-01-01: >250 trading days before the first event)")
    parser.add_argument("--end", default=(date.today() + timedelta(days=1)).isoformat(),
                        help="last date, exclusive (default: tomorrow, i.e. include today)")
    parser.add_argument("--out", default="data/raw", help="output folder")
    parser.add_argument("--pause", type=float, default=1.0,
                        help="seconds between tickers, to be gentle with Yahoo")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    frames, log = [], {
        "run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "start": args.start, "end_exclusive": args.end,
        "python": sys.version.split()[0], "pandas": pd.__version__,
        "yfinance": getattr(yf, "__version__", "unknown"),
        "tickers": {}, "failed": [],
    }

    print(f"Downloading {len(TICKERS)} tickers from {args.start} to {args.end} (exclusive)")
    for meta in TICKERS:
        t = meta["ticker"]
        print(f"- {t:<10} {meta['name']}")
        try:
            df = download_one(t, args.start, args.end)
        except RuntimeError as exc:
            print(f"  FAILED: {exc}")
            log["failed"].append(t)
            continue
        frames.append(df)
        log["tickers"][t] = quality_report(df)
        time.sleep(args.pause)

    if not frames:
        sys.exit("No data downloaded. Check your internet connection and try again.")

    prices = pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"])
    prices["date"] = prices["date"].dt.strftime("%Y-%m-%d")
    prices.to_csv(out / "prices_long.csv", index=False)
    pd.DataFrame(TICKERS).to_csv(out / "tickers_meta.csv", index=False)
    (out / "download_log.json").write_text(json.dumps(log, indent=2))

    # ---- Summary for the console --------------------------------------------
    print("\nSummary")
    print(f"{'ticker':<10} {'rows':>6}  {'first':<10}  {'last':<10}  flags")
    for t, r in log["tickers"].items():
        flags = []
        if r["large_moves"]:
            flags.append(f"{len(r['large_moves'])} moves >25%")
        if r["calendar_gaps_over_7_days"]:
            flags.append(f"{len(r['calendar_gaps_over_7_days'])} gaps >7d")
        if r["non_positive_prices"]:
            flags.append("non-positive prices")
        print(f"{t:<10} {r['rows']:>6}  {r['first_date']:<10}  {r['last_date']:<10}  {', '.join(flags) or 'ok'}")
    if log["failed"]:
        print(f"\nFailed tickers: {', '.join(log['failed'])} -- rerun the script later")
    print(f"\nSaved {len(prices):,} rows to {out / 'prices_long.csv'}")
    print("Upload prices_long.csv and download_log.json for the next step.")


if __name__ == "__main__":
    main()
