"""
Export the results of event_study.py into docs/data.js for the online dashboard.

Usage (from the repository root, after event_study.py):
    python scripts/build_dashboard.py

The dashboard (docs/index.html) is a static page: GitHub Pages serves the
docs/ folder as it is, and the page also opens offline by double-clicking it.
The data is written as a JavaScript file (window.DASH = {...}) instead of JSON
so that it loads even from the local disk, where browsers block fetch().
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
DOCS = ROOT / "docs"
GROUPS = ["equipment", "manufacturer", "designer"]
SEVERE_CONFOUNDERS = {"E16", "E19", "L01"}   # keep in sync with event_study.py


def r(x, d=4):
    return None if pd.isna(x) else round(float(x), d)


def main():
    DOCS.mkdir(exist_ok=True)
    events = pd.read_csv(ROOT / "data" / "events.csv")
    car = pd.read_csv(RES / "car_stock_event.csv")
    ar = pd.read_csv(RES / "ar_daily.csv")
    tests = pd.read_csv(RES / "group_tests.csv")
    target = pd.read_csv(RES / "targeting_test.csv")
    rob = pd.read_csv(RES / "robustness.csv")
    meta = pd.read_csv(ROOT / "data" / "raw" / "tickers_meta.csv")

    companies = (car[["ticker", "name", "group"]].drop_duplicates()
                 .merge(meta[["ticker", "country", "exchange"]], on="ticker"))
    companies["o"] = companies["group"].map({g: i for i, g in enumerate(GROUPS)})
    companies = companies.sort_values(["o", "name"]).drop(columns="o")

    # cumulative AR from day -10, per stock-event
    ar = ar.merge(companies[["ticker", "group"]], on="ticker").sort_values("rel_day")
    ar["cum"] = ar.groupby(["event_id", "ticker"])["ar"].cumsum()
    grp_path = ar.groupby(["event_id", "group", "rel_day"])["cum"].mean()

    ev_out = []
    for e in events.itertuples():
        c = car[(car["event_id"] == e.event_id)]
        cars = {}
        for w in c["window"].unique():
            cw = c[c["window"] == w].set_index("ticker")
            cars[w] = {t: [r(cw.at[t, "car"]), r(cw.at[t, "p_value"], 3)] for t in cw.index}
        paths = {g: [r(v) for v in grp_path.loc[(e.event_id, g)].values]
                 for g in GROUPS if (e.event_id, g) in grp_path.index}
        ev_out.append(dict(
            id=e.event_id, date=e.date, country=e.country, issuer=e.issuer, title=e.title,
            short=e.short_label, direction=e.direction, sample=e.sample,
            target=e.most_exposed_group, timing=e.timing_note, confounders=e.confounders,
            confounded=e.event_id in SEVERE_CONFOUNDERS,
            source=e.source_primary, source2=e.source_secondary,
            car=cars, path=paths))

    # average path by group across events, for two samples
    def avg_path(ids):
        a = ar[ar["event_id"].isin(ids)]
        p = (a.groupby(["group", "event_id", "rel_day"])["cum"].mean()
               .groupby(["group", "rel_day"]).agg(["mean", "std", "count"]))
        out = {}
        for g in GROUPS:
            d = p.loc[g]
            se = d["std"] / np.sqrt(d["count"])
            out[g] = dict(mean=[r(v) for v in d["mean"]],
                          lo=[r(v) for v in d["mean"] - 1.96 * se],
                          hi=[r(v) for v in d["mean"] + 1.96 * se])
        return out

    main_ids = events.loc[events["sample"] == "main", "event_id"]
    clean_ids = main_ids[~main_ids.isin(SEVERE_CONFOUNDERS)]

    t = tests[tests["window"] == "CAR[0,+1]"]
    group_tests = {s: {row.group: dict(mean=r(row.mean_car), p=r(row.p_ttest, 3), n=int(row.n_events))
                       for row in t[t["sample"] == s].itertuples()}
                   for s in t["sample"].unique()}
    tg = target.set_index("window")

    data = dict(
        generated=pd.Timestamp.today().strftime("%Y-%m-%d"),
        groups=GROUPS,
        companies=companies.to_dict("records"),
        events=ev_out,
        paths={"main": avg_path(main_ids), "clean": avg_path(clean_ids)},
        n_events={"main": int(len(main_ids)), "clean": int(len(clean_ids))},
        group_tests=group_tests,
        targeting={w: dict(targeted=r(tg.at[w, "mean_targeted"]), others=r(tg.at[w, "mean_others"]),
                           diff=r(tg.at[w, "mean_diff"]), p=r(tg.at[w, "p_ttest"], 3),
                           n=int(tg.at[w, "n_events"])) for w in tg.index},
        robustness=json.loads(rob.round(4).to_json(orient="records")),
    )
    (DOCS / "data.js").write_text("window.DASH = " + json.dumps(data, separators=(",", ":")) + ";\n",
                                  encoding="utf-8")
    print(f"Wrote {DOCS / 'data.js'}  ({len(ev_out)} events, {len(companies)} companies)")


if __name__ == "__main__":
    main()
