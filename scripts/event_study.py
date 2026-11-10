"""
Event study: how semiconductor stocks react to export-control announcements.

Usage (from the repository root, after download_prices.py):
    python scripts/event_study.py

Inputs:
    data/events.csv             announcements (date, time, time zone, sample, ...)
    data/raw/prices_long.csv    daily prices from download_prices.py
    data/raw/tickers_meta.csv   role in the supply chain and benchmark index

Outputs (in results/):
    ar_daily.csv            abnormal return of every stock, every event, days -10..+10
    car_stock_event.csv     cumulative abnormal return (CAR) per stock, event and window
    group_tests.csv         average reaction of each group and its statistical tests
    group_differences.csv   do the groups react differently from each other?
    targeting_test.csv      does the targeted group react more than the others?
    robustness.csv          headline results under alternative choices
    fig_caar_by_group*.png  average cumulative reaction by group, days -10..+10
    fig_heatmap_car01.png   CAR[0,+1] for every event and stock

Method in short (details in the README):
    1. Market model estimated on 250 trading days ending 11 days before the event:
           R_stock = alpha + beta * R_market + error
       Days around the other announcements are left out of the estimation.
    2. Abnormal return (AR) = actual return - (alpha + beta * market return).
    3. CAR = sum of ARs over a window, e.g. [0,+1] = announcement day + next day.
    4. Tests. Stocks react on the same day, so their errors are correlated and
       cannot be treated as independent observations. The main test therefore
       averages the stocks of a group within each event and then tests across
       events (N = number of events). Stock-level tests are reported too.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Settings: every methodological choice is here, in one place
# ---------------------------------------------------------------------------
EST_LENGTH = 250        # trading days in the estimation window (about one year)
EST_GAP = 10            # days between estimation window and event: [-260, -11]
MIN_EST_OBS = 120       # skip a stock-event if fewer clean estimation days remain
EXCLUDE_AROUND_OTHER_EVENTS = (-1, 5)   # days removed around every other event
PLOT_WINDOW = (-10, 10)

WINDOWS = {             # (first day, last day), relative to day 0
    "CAR[0,+1]": (0, 1),     # main window: robust to unknown announcement time
    "CAR[-1,+1]": (-1, 1),
    "CAR[0,+5]": (0, 5),     # does the reaction persist or reverse?
    "CAR[-5,-1]": (-5, -1),  # leakage / anticipation before the announcement
}
MAIN_WINDOW = "CAR[0,+1]"

GROUPS = ["equipment", "manufacturer", "designer"]

# Events with a major unrelated news item inside the [0,+1] window
SEVERE_CONFOUNDERS = {"E16", "E19", "L01"}

# Closing time of each exchange, used to find day 0 for every stock
EXCHANGE_CLOSE = {
    "Nasdaq": ("16:00", "America/New_York"),
    "NYSE": ("16:00", "America/New_York"),
    "US": ("16:00", "America/New_York"),
    "Euronext Amsterdam": ("17:30", "Europe/Amsterdam"),
    "Europe": ("17:30", "Europe/Amsterdam"),
    "Tokyo": ("15:00", "Asia/Tokyo"),
    "Taiwan": ("13:30", "Asia/Taipei"),
    "Korea": ("15:30", "Asia/Seoul"),
    "Hong Kong": ("16:00", "Asia/Hong_Kong"),
    "Shanghai": ("15:00", "Asia/Shanghai"),
}

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
def load():
    events = pd.read_csv(DATA / "events.csv", parse_dates=["date"])
    prices = pd.read_csv(DATA / "raw" / "prices_long.csv", parse_dates=["date"])
    meta = pd.read_csv(DATA / "raw" / "tickers_meta.csv")
    # Timestamp of each announcement in UTC
    local = pd.to_datetime(events["date"].dt.strftime("%Y-%m-%d") + " " + events["announce_time"])
    events["announce_utc"] = [
        t.tz_localize(z).tz_convert("UTC") for t, z in zip(local, events["announce_tz"])
    ]
    wide = prices.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    return events, wide, meta


def stock_returns(wide: pd.DataFrame, stock: str, bench: str) -> pd.DataFrame:
    """Daily returns of a stock and its benchmark on the days BOTH traded.

    Using common days means each return covers exactly the same period
    for the stock and for its index (holidays differ across markets).
    """
    px = wide[[stock, bench]].dropna()
    ret = px.pct_change().dropna()
    ret.columns = ["r", "rm"]
    return ret


# ---------------------------------------------------------------------------
# 2. Day 0: first session of the stock's exchange that closes after the news
# ---------------------------------------------------------------------------
def session_close_utc(dates: pd.DatetimeIndex, exchange: str) -> pd.Series:
    hhmm, tz = EXCHANGE_CLOSE[exchange]
    local = pd.to_datetime(dates.strftime("%Y-%m-%d") + " " + hhmm)
    closes = local.tz_localize(tz).tz_convert("UTC")
    return pd.Series(closes, index=dates)


def find_day0(trading_days: pd.DatetimeIndex, exchange: str, announce_utc) -> int | None:
    closes = session_close_utc(trading_days, exchange)
    after = np.flatnonzero((closes > announce_utc).to_numpy())
    return int(after[0]) if len(after) else None


# ---------------------------------------------------------------------------
# 3. Market model and abnormal returns for one stock and one event
# ---------------------------------------------------------------------------
def market_model(ret: pd.DataFrame, d0: int, other_day0s: list[int],
                 est_length: int = EST_LENGTH, model: str = "market"):
    """model="market": OLS alpha and beta.  model="adjusted": alpha=0, beta=1
    (market-adjusted returns, no estimation: a robustness check)."""
    start, end = d0 - EST_GAP - est_length, d0 - EST_GAP  # positions [start, end)
    if start < 0:
        return None
    est_pos = np.arange(start, end)
    lo, hi = EXCLUDE_AROUND_OTHER_EVENTS
    dirty = set()
    for o in other_day0s:
        dirty.update(range(o + lo, o + hi + 1))
    est_pos = np.array([p for p in est_pos if p not in dirty])
    if len(est_pos) < min(MIN_EST_OBS, int(0.8 * est_length)):
        return None
    est = ret.iloc[est_pos]
    if model == "adjusted":
        alpha, beta = 0.0, 1.0
    else:
        beta, alpha = np.polyfit(est["rm"], est["r"], 1)
    resid = est["r"] - (alpha + beta * est["rm"])
    sigma = resid.std(ddof=2)
    return dict(alpha=alpha, beta=beta, sigma=sigma, n_est=len(est_pos))


def run_stock_event(ret, d0, mm, event_id, ticker):
    lo, hi = PLOT_WINDOW
    pos = np.arange(d0 + lo, d0 + hi + 1)
    pos = pos[(pos >= 0) & (pos < len(ret))]
    win = ret.iloc[pos].copy()
    win["ar"] = win["r"] - (mm["alpha"] + mm["beta"] * win["rm"])
    win["rel_day"] = pos - d0
    win["event_id"], win["ticker"] = event_id, ticker
    return win.reset_index()


# ---------------------------------------------------------------------------
# 4. Statistical tests
# ---------------------------------------------------------------------------
def one_sample(x: pd.Series) -> dict:
    x = x.dropna()
    n = len(x)
    out = dict(n=n, mean=x.mean() if n else np.nan, median=x.median() if n else np.nan,
               share_negative=(x < 0).mean() if n else np.nan)
    if n >= 3:
        t, p = stats.ttest_1samp(x, 0.0)
        out.update(t_stat=t, p_ttest=p)
        try:
            out["p_wilcoxon"] = stats.wilcoxon(x).pvalue   # does not assume normality
        except ValueError:
            out["p_wilcoxon"] = np.nan
    else:
        out.update(t_stat=np.nan, p_ttest=np.nan, p_wilcoxon=np.nan)
    return out


def stars(p):
    return "" if pd.isna(p) else "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def compute_car(events, wide, stocks, est_length=EST_LENGTH, model="market"):
    """AR and CAR for every stock-event pair."""
    ar_rows, car_rows = [], []
    for _, s in stocks.iterrows():
        ret = stock_returns(wide, s["ticker"], s["benchmark"])
        days = ret.index
        day0 = {e.event_id: find_day0(days, s["exchange"], e.announce_utc) for e in events.itertuples()}
        for e in events.itertuples():
            d0 = day0[e.event_id]
            if d0 is None or d0 + max(w[1] for w in WINDOWS.values()) >= len(ret):
                continue
            others = [v for k, v in day0.items() if k != e.event_id and v is not None]
            mm = market_model(ret, d0, others, est_length, model)
            if mm is None:
                continue
            win = run_stock_event(ret, d0, mm, e.event_id, s["ticker"])
            ar_rows.append(win)
            for wname, (a, b) in WINDOWS.items():
                ars = win.loc[win["rel_day"].between(a, b), "ar"]
                L = len(ars)
                car = ars.sum()
                t = car / (mm["sigma"] * np.sqrt(L))
                car_rows.append(dict(
                    event_id=e.event_id, ticker=s["ticker"], name=s["name"], group=s["role"],
                    window=wname, car=car, t_stat=t,
                    p_value=2 * stats.t.sf(abs(t), df=mm["n_est"] - 2),
                    scar=t,  # standardized CAR, used by the BMP test
                    day0=win.loc[win["rel_day"] == 0, "date"].iloc[0].date(),
                    **mm))
    ar = pd.concat(ar_rows, ignore_index=True)
    car = pd.DataFrame(car_rows).merge(
        events[["event_id", "date", "title", "country", "direction", "sample", "most_exposed_group"]],
        on="event_id")
    return ar, car


def portfolios(car, events):
    """Event-level group portfolios: average CAR of the stocks of a group, per event."""
    port = (car.groupby(["event_id", "window", "group"])["car"].mean()
               .unstack("group").reset_index())
    allg = car.groupby(["event_id", "window"])["car"].mean().rename("all").reset_index()
    port = port.merge(allg, on=["event_id", "window"])
    return port.merge(events[["event_id", "sample", "direction", "most_exposed_group"]], on="event_id")


def targeting_test(port, wname):
    """Targeted group minus the average of the other two groups, per event."""
    p = port[(port["window"] == wname) & (port["sample"] == "main")
             & port["most_exposed_group"].isin(GROUPS)].copy()
    p["targeted"] = [r[r["most_exposed_group"]] for _, r in p.iterrows()]
    p["others"] = [np.nanmean([r[g] for g in GROUPS if g != r["most_exposed_group"]])
                   for _, r in p.iterrows()]
    res = one_sample(p["targeted"] - p["others"])
    return dict(window=wname, n_events=res["n"], mean_targeted=p["targeted"].mean(),
                mean_others=p["others"].mean(), mean_diff=res["mean"],
                t_stat=res["t_stat"], p_ttest=res["p_ttest"], p_wilcoxon=res["p_wilcoxon"],
                events=", ".join(p["event_id"]))


def main():
    OUT.mkdir(exist_ok=True)
    events, wide, meta = load()
    stocks = meta[meta["sample"] == "main"]
    ar, car = compute_car(events, wide, stocks)
    ar.to_csv(OUT / "ar_daily.csv", index=False)
    car.to_csv(OUT / "car_stock_event.csv", index=False)

    # -------- Samples -----------------------------------------------------
    samples = {
        "main (tightening)": car["sample"] == "main",
        "main excl. confounded": (car["sample"] == "main") & ~car["event_id"].isin(SEVERE_CONFOUNDERS),
        "all tightening incl. robustness": car["direction"] == "tightening",
        "loosening": car["sample"] == "loosening",
    }

    port = portfolios(car, events)

    # -------- Group tests ---------------------------------------------------
    rows = []
    for sname, mask in samples.items():
        ev_ids = car.loc[mask, "event_id"].unique()
        for wname in WINDOWS:
            p = port[(port["window"] == wname) & port["event_id"].isin(ev_ids)]
            c = car[mask & (car["window"] == wname)]
            for g in GROUPS + ["all"]:
                res = one_sample(p[g])  # event-level test (main)
                cg = c if g == "all" else c[c["group"] == g]
                bmp = one_sample(cg["scar"])  # stock-level BMP test
                rows.append(dict(sample=sname, window=wname, group=g, n_events=res["n"],
                                 mean_car=res["mean"], median_car=res["median"],
                                 share_negative=res["share_negative"], t_stat=res["t_stat"],
                                 p_ttest=res["p_ttest"], p_wilcoxon=res["p_wilcoxon"],
                                 n_stock_events=bmp["n"], bmp_t=bmp["t_stat"], bmp_p=bmp["p_ttest"]))
    group_tests = pd.DataFrame(rows)
    group_tests.to_csv(OUT / "group_tests.csv", index=False)

    # -------- Group differences (paired across events) ---------------------
    rows = []
    pairs = [("equipment", "designer"), ("equipment", "manufacturer"), ("manufacturer", "designer")]
    for sname, mask in samples.items():
        ev_ids = car.loc[mask, "event_id"].unique()
        for wname in WINDOWS:
            p = port[(port["window"] == wname) & port["event_id"].isin(ev_ids)]
            for a, b in pairs:
                d = (p[a] - p[b]).dropna()
                res = one_sample(d)
                rows.append(dict(sample=sname, window=wname, comparison=f"{a} - {b}",
                                 n_events=res["n"], mean_diff=res["mean"], t_stat=res["t_stat"],
                                 p_ttest=res["p_ttest"], p_wilcoxon=res["p_wilcoxon"]))
    group_diff = pd.DataFrame(rows)
    group_diff.to_csv(OUT / "group_differences.csv", index=False)

    # -------- Targeting: does the targeted group fall more? ------------------
    targeting = pd.DataFrame([targeting_test(port, w) for w in WINDOWS])
    targeting.to_csv(OUT / "targeting_test.csv", index=False)

    # -------- Robustness: do the headline results survive other choices? -----
    robustness = run_robustness(events, wide, meta)
    robustness.to_csv(OUT / "robustness.csv", index=False)

    make_figures(ar, car, events)
    print_summary(group_tests, group_diff, targeting, car)
    print_robustness(robustness)


# ---------------------------------------------------------------------------
# Robustness checks
# ---------------------------------------------------------------------------
def headline(car, events, wname=MAIN_WINDOW):
    """The three headline numbers, recomputed for one specification."""
    port = portfolios(car, events)
    p = port[port["window"] == wname]
    clean = events.loc[(events["sample"] == "main")
                       & ~events["event_id"].isin(SEVERE_CONFOUNDERS), "event_id"]
    main_ids = events.loc[events["sample"] == "main", "event_id"]
    all_main = one_sample(p.loc[p["event_id"].isin(main_ids), "all"])
    all_clean = one_sample(p.loc[p["event_id"].isin(clean), "all"])
    des_clean = one_sample(p.loc[p["event_id"].isin(clean), "designer"])
    tgt = targeting_test(port, wname)
    return {
        "all stocks, main: mean": all_main["mean"], "all stocks, main: p": all_main["p_ttest"],
        "all stocks, excl. confounded: mean": all_clean["mean"],
        "all stocks, excl. confounded: p": all_clean["p_ttest"],
        "designers, excl. confounded: mean": des_clean["mean"],
        "designers, excl. confounded: p": des_clean["p_ttest"],
        "targeted minus others: diff": tgt["mean_diff"], "targeted minus others: p": tgt["p_ttest"],
    }


def run_robustness(events, wide, meta):
    main = meta[meta["sample"] == "main"]
    adr = main.copy()
    for local, us in {"ASML.AS": "ASML", "2330.TW": "TSM"}.items():
        adr.loc[adr["ticker"] == local, ["ticker", "exchange", "benchmark", "name"]] = \
            meta.loc[meta["ticker"] == us, ["ticker", "exchange", "benchmark", "name"]].values
    specs = {
        "1. Base: market model, 250 days, CAR[0,+1]": dict(stocks=main),
        "2. Market-adjusted returns (beta = 1)": dict(stocks=main, model="adjusted"),
        "3. Shorter estimation window (120 days)": dict(stocks=main, est_length=120),
        "4. ASML and TSMC via New York listings": dict(stocks=adr),
        "5. Without SMIC": dict(stocks=main[main["ticker"] != "0981.HK"]),
        "6. Wider window CAR[-1,+1]": dict(stocks=main, window="CAR[-1,+1]"),
        "7. Longer window CAR[0,+5]": dict(stocks=main, window="CAR[0,+5]"),
    }
    rows = []
    for name, sp in specs.items():
        _, car = compute_car(events, wide, sp["stocks"],
                             est_length=sp.get("est_length", EST_LENGTH),
                             model=sp.get("model", "market"))
        rows.append({"specification": name, **headline(car, events, sp.get("window", MAIN_WINDOW))})
    return pd.DataFrame(rows)


def print_robustness(rob):
    print("\nRobustness of the headline results (mean in %, p-value in brackets)")
    print(f"{'specification':<46}{'all, main':>16}{'all, clean':>16}{'designers, clean':>18}{'targeted - others':>19}")
    for _, r in rob.iterrows():
        cells = []
        for k in ["all stocks, main", "all stocks, excl. confounded",
                  "designers, excl. confounded", "targeted minus others"]:
            m = r[f"{k}: mean"] if f"{k}: mean" in r else r[f"{k}: diff"]
            cells.append(f"{m*100:6.2f} ({r[f'{k}: p']:.2f})")
        print(f"{r['specification']:<46}{cells[0]:>16}{cells[1]:>16}{cells[2]:>18}{cells[3]:>19}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def make_figures(ar, car, events):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"equipment": "#2a6f97", "manufacturer": "#c8553d", "designer": "#588157"}
    groups = car[["ticker", "group"]].drop_duplicates()
    figure_samples = {
        "fig_caar_by_group.png": ("main sample", events["sample"] == "main"),
        "fig_caar_by_group_excl_confounded.png": (
            "main sample excl. confounded events",
            (events["sample"] == "main") & ~events["event_id"].isin(SEVERE_CONFOUNDERS)),
    }
    for fname, (label, mask) in figure_samples.items():
        a = ar[ar["event_id"].isin(events.loc[mask, "event_id"])].merge(groups, on="ticker")
        a = a.sort_values("rel_day")
        # cumulate from day -10, average within group-event, then across events
        a["cum"] = a.groupby(["event_id", "ticker"])["ar"].cumsum()
        path = (a.groupby(["group", "event_id", "rel_day"])["cum"].mean()
                  .groupby(["group", "rel_day"]).agg(["mean", "std", "count"]).reset_index())
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for g in GROUPS:
            d = path[path["group"] == g]
            se = d["std"] / np.sqrt(d["count"])
            members = ", ".join(sorted(car.loc[car["group"] == g, "name"].unique()))
            ax.plot(d["rel_day"], d["mean"] * 100, color=colors[g], lw=2,
                    label=f"{g}: {members}")
            ax.fill_between(d["rel_day"], (d["mean"] - 1.96 * se) * 100,
                            (d["mean"] + 1.96 * se) * 100, color=colors[g], alpha=0.12)
        ax.axvline(0, color="grey", ls="--", lw=1)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xticks(range(PLOT_WINDOW[0], PLOT_WINDOW[1] + 1, 2))
        ax.set_xlabel("Trading days relative to the announcement (day 0)")
        ax.set_ylabel("Cumulative abnormal return (%)")
        ax.set_title(f"Average cumulative abnormal return by group, {label}\n"
                     f"(shaded band = 95% confidence interval across {d['count'].max()} events)",
                     fontsize=11)
        ax.legend(frameon=False, fontsize=8, loc="lower left")
        fig.tight_layout()
        fig.savefig(OUT / fname, dpi=150)
        plt.close(fig)

    # Heatmap: rows = events (chronological, tightening then loosening),
    # columns = companies grouped by role in the supply chain
    names = car[["ticker", "name", "group"]].drop_duplicates()
    names["o"] = names["group"].map({g: i for i, g in enumerate(GROUPS)})
    names = names.sort_values(["o", "name"])
    ev = events.copy()
    ev["block"] = (ev["direction"] == "loosening").astype(int)
    ev = ev.sort_values(["block", "date"])
    ev["label"] = [f"{r.date:%Y-%m-%d}  {r.short_label}  ({r.event_id})" for r in ev.itertuples()]
    h = (car[car["window"] == MAIN_WINDOW].pivot(index="event_id", columns="ticker", values="car") * 100)
    h = h.reindex(index=ev["event_id"], columns=names["ticker"])
    lim = np.nanpercentile(np.abs(h.values), 95)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(h.values, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(names)), names["name"], rotation=40, ha="right")
    ax.set_yticks(range(len(ev)), ev["label"])
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            v = h.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(v) > 0.6 * lim else "black")
    # separators: between supply-chain groups and between tightening / loosening
    for k in np.flatnonzero(np.diff(names["o"].values)) + 0.5:
        ax.axvline(k, color="black", lw=1.5)
    for k in np.flatnonzero(np.diff(ev["block"].values)) + 0.5:
        ax.axhline(k, color="black", lw=1.5)
    for g in GROUPS:  # group names above the columns
        cols = np.flatnonzero(names["group"].values == g)
        ax.text(cols.mean(), -0.9, g, ha="center", va="bottom", fontsize=11, fontweight="bold")
    for confounded in SEVERE_CONFOUNDERS:  # mark events with major unrelated news
        if confounded in list(ev["event_id"]):
            ax.get_yticklabels()[list(ev["event_id"]).index(confounded)].set_color("grey")
    fig.colorbar(im, ax=ax, label="CAR[0,+1] in % (red = fell more than expected)", shrink=0.6)
    ax.set_title("Abnormal return over announcement day + next day, by event and company\n"
                 "top block: tightening   |   bottom block: loosening   |   grey labels: confounded events",
                 pad=28)
    fig.tight_layout()
    fig.savefig(OUT / "fig_heatmap_car01.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
def print_summary(group_tests, group_diff, targeting, car):
    pd.set_option("display.width", 140)
    print(f"\nStock-event pairs estimated: {car[['event_id', 'ticker']].drop_duplicates().shape[0]}")
    for sname in group_tests["sample"].unique():
        g = group_tests[(group_tests["sample"] == sname) & (group_tests["window"] == MAIN_WINDOW)]
        print(f"\n{MAIN_WINDOW} - {sname}  (event-level test; N = events)")
        for _, r in g.iterrows():
            print(f"  {r['group']:<13} mean {r['mean_car']*100:6.2f}%  t={r['t_stat']:5.2f}  "
                  f"p={r['p_ttest']:.3f}{stars(r['p_ttest']):<3}  Wilcoxon p={r['p_wilcoxon']:.3f}  "
                  f"N={r['n_events']}")
    d = group_diff[(group_diff["sample"] == "main (tightening)") & (group_diff["window"] == MAIN_WINDOW)]
    print(f"\nGroup differences, {MAIN_WINDOW}, main sample")
    for _, r in d.iterrows():
        print(f"  {r['comparison']:<26} {r['mean_diff']*100:6.2f}%  p={r['p_ttest']:.3f}{stars(r['p_ttest'])}")
    t = targeting[targeting["window"] == MAIN_WINDOW].iloc[0]
    print(f"\nTargeted group vs others, {MAIN_WINDOW}: targeted {t['mean_targeted']*100:.2f}%, "
          f"others {t['mean_others']*100:.2f}%, diff p={t['p_ttest']:.3f} (N={t['n_events']})")
    print(f"\nResults saved in {OUT}")


if __name__ == "__main__":
    main()
