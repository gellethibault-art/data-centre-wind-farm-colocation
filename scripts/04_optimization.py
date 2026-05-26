"""
04_optimization.py — Operational Optimization of DC Curtailment Absorption
Group 10: Wind Curtailment and Data Centres

Moves beyond the simple "greedy" min(curtailment, flex) matching in script 03
by formulating a per-day linear programme (LP) that respects real-world
operational constraints:

  1. Deliverability — fraction of curtailed power that can actually reach the
     DC (accounts for transmission constraints if not co-located).
  2. Ramp rate — maximum change in flexible load per settlement period (MW/SP).
  3. Backlog budget — how much flexible compute work is actually available to
     schedule on a given day (as a fraction of total available flex energy).

The LP maximises absorbed energy subject to these constraints, providing a
more realistic bound on what DC flexibility can achieve.

Approach follows the methodology in:
  Ahmadi, Knorr & Meschede (2025) — "Improvement of wind power utilisation
    through flexible operation of data center in wind parks," Renewable Energy.
  Lin, Zavala & Chien (2021) — "Evaluating coupling models for cloud
    datacenters and power grids," ACM e-Energy.

Input:  output/csv/03_matched_timeseries.csv      (from 03, Part B)
    OR  data/processed/curtailment_per_period.csv  (fallback to profile-based)
        data/processed/dc_hourly_profile_matched.csv
        data/processed/curtailment_processed.csv   (for farm grouping)
Output: output/csv/04_summary_results.csv
        output/csv/04_waterfall.csv
        output/charts/04_*.png
"""

from __future__ import annotations
import os, sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# ── paths & style ─────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR  = os.path.join(ROOT, "data", "processed")
CSV_DIR   = os.path.join(ROOT, "output", "csv")
CHART_DIR = os.path.join(ROOT, "output", "charts")
for d in [CSV_DIR, CHART_DIR]:
    os.makedirs(d, exist_ok=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import apply_style, COLORS, SCENARIO_COLORS, save_fig
apply_style()

PERIOD_HOURS = 0.5  # 30-min settlement period

# ── input paths ───────────────────────────────────────────────────────────────
TS_PATH         = os.path.join(CSV_DIR, "03_matched_timeseries.csv")
CURT_PERIOD_PATH = os.path.join(PROC_DIR, "curtailment_per_period.csv")
DC_HOURLY_PATH   = os.path.join(PROC_DIR, "dc_hourly_profile_matched.csv")
CURT_PROC_PATH   = os.path.join(PROC_DIR, "curtailment_processed.csv")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — edit these to change scenarios (no interactive input needed)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Scenario:
    name: str
    fleet_mw: float
    flex_frac: float        # fraction of headroom that is dispatchable
    deliverability: float   # 1.0 = co-located, <1.0 = remote
    backlog_frac: float     # 1.0 = unlimited backlog of flex work
    ramp_frac: float        # max Δx per SP as fraction of fleet MW

SCENARIOS: List[Scenario] = [
    # Waterfall: progressively add realism to a 1 GW fleet
    Scenario("Ideal (no constraints)",                       1000, 0.60, 1.0, 1.00, 1.00),
    Scenario("+ Deliverability (60%)",                       1000, 0.60, 0.6, 1.00, 1.00),
    Scenario("+ Backlog limit (50%)",                        1000, 0.60, 0.6, 0.50, 1.00),
    Scenario("+ Ramp limit (50% fleet/SP)",                  1000, 0.60, 0.6, 0.50, 0.50),
    Scenario("+ Tight ramp (10% fleet/SP)",                  1000, 0.60, 0.6, 0.50, 0.10),
]

# Optional: farm grouping — set to None to skip, or specify farm names
# Example: FARM_GROUP = ("scottish_offshore", ["Seagreen 1", "Seagreen 2"])
FARM_GROUP: Optional[Tuple[str, List[str]]] = None


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_timeseries() -> pd.DataFrame:
    """Load the best available time series — prefer the real half-hourly
    series from Part B of script 03; fall back to profile-based construction."""

    if os.path.exists(TS_PATH):
        print(f"  Using real half-hourly time series: {TS_PATH}")
        df = pd.read_csv(TS_PATH, parse_dates=["Datetime"])
        df = df.sort_values("Datetime").set_index("Datetime")

        full_idx = pd.date_range(df.index.min(), df.index.max(), freq="30min")
        df = df.reindex(full_idx)
        df["curtailment_MW"]  = df["curtailment_MW"].fillna(0.0).clip(lower=0.0)
        df["curtailment_MWh"] = df["curtailment_MW"] * PERIOD_HOURS

        if df["dc_headroom_pct"].isna().any():
            hour = df.index.hour
            hour_med = df.groupby(hour)["dc_headroom_pct"].median()
            df["dc_headroom_pct"] = df["dc_headroom_pct"].fillna(
                pd.Series(hour, index=df.index).map(hour_med))
        df["dc_headroom_pct"] = df["dc_headroom_pct"].clip(0, 100)

    elif os.path.exists(CURT_PERIOD_PATH) and os.path.exists(DC_HOURLY_PATH):
        print(f"  No real time series found — building from profiles (profile-based fallback)")
        cur = pd.read_csv(CURT_PERIOD_PATH)
        cur["Datetime"] = pd.to_datetime(cur["Datetime"] if "Datetime" in cur.columns else cur["Date"])
        cur = cur.sort_values("Datetime").set_index("Datetime")

        full_idx = pd.date_range(cur.index.min(), cur.index.max(), freq="30min")

        if "curtailment_MW" in cur.columns:
            curt_mw = cur.groupby(cur.index)["curtailment_MW"].sum()
        else:
            curt_mw = cur.groupby(cur.index)["total_curtailment_MWh"].sum() / 0.5
        curt_mw = curt_mw.reindex(full_idx, fill_value=0.0)

        dc = pd.read_csv(DC_HOURLY_PATH, index_col=0)
        dc.index = dc.index.astype(int)
        headroom_map = (100 - dc["mean_util"]).to_dict()

        headroom_pct = pd.Series(full_idx.hour, index=full_idx).map(headroom_map).clip(0, 100)

        df = pd.DataFrame({
            "curtailment_MW":   curt_mw.values,
            "curtailment_MWh":  curt_mw.values * PERIOD_HOURS,
            "dc_headroom_pct":  headroom_pct.values,
        }, index=full_idx)
    else:
        sys.exit("ERROR: Need either 03_matched_timeseries.csv or curtailment_per_period.csv + dc_hourly_profile_matched.csv")

    df["date"] = df.index.date
    df["hour"] = df.index.hour
    return df


def build_farm_ts(farm_names: List[str], general_ts: pd.DataFrame) -> pd.DataFrame:
    """Build time series filtered to specific wind farms."""
    if not os.path.exists(CURT_PROC_PATH):
        raise FileNotFoundError(f"Need {CURT_PROC_PATH} for farm analysis. Run 01 first.")

    sp = pd.read_csv(CURT_PROC_PATH)
    sp["Datetime"] = pd.to_datetime(sp["Datetime"], errors="coerce")
    sp = sp.dropna(subset=["Datetime"])

    mask = sp["Generator_Full_Name"].isin(farm_names)
    curt = sp.loc[mask].groupby("Datetime")["Curtailment_MWh"].sum()
    curt_mw = curt / PERIOD_HOURS

    result = general_ts.copy()
    result["curtailment_MW"]  = curt_mw.reindex(result.index, fill_value=0.0).values
    result["curtailment_MWh"] = result["curtailment_MW"] * PERIOD_HOURS
    return result


# ══════════════════════════════════════════════════════════════════════════════
# OPTIMIZATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def greedy_dispatch(df: pd.DataFrame, sc: Scenario) -> np.ndarray:
    """Per-period greedy: absorbed = min(deliverability × curtailment, flex capacity)."""
    cur_eff = sc.deliverability * df["curtailment_MW"].to_numpy()
    avail   = sc.fleet_mw * (df["dc_headroom_pct"].to_numpy() / 100.0) * sc.flex_frac
    return np.minimum(cur_eff, avail)


def optimize_one_day(cur_eff: np.ndarray, avail: np.ndarray, sc: Scenario) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-day LP to maximise absorbed curtailment under operational constraints.

    Decision variables (size 2n):
      x_t = dispatched flexible load (MW),  t = 0..n-1
      y_t = absorbed curtailment (MW),      t = n..2n-1

    Constraints:
      0 ≤ x_t ≤ avail_t                    (can't exceed available headroom)
      0 ≤ y_t ≤ cur_eff_t                  (can't absorb more than delivered)
      y_t ≤ x_t                            (must be running to absorb)
      Σ x_t · 0.5 ≤ backlog · Σ avail_t · 0.5  (daily energy budget)
      |x_t − x_{t−1}| ≤ ramp_mw           (ramp rate limit)
    """
    n = len(cur_eff)
    if n == 0:
        return np.array([]), np.array([])

    m = 2 * n
    e_budget  = sc.backlog_frac * avail.sum() * PERIOD_HOURS
    ramp_mw   = sc.ramp_frac * sc.fleet_mw

    # Objective: minimise −Σy_t  (= maximise absorption)
    c = np.zeros(m)
    c[n:] = -1.0

    bounds = ([(0.0, float(avail[t])) for t in range(n)] +
              [(0.0, float(cur_eff[t])) for t in range(n)])

    A_rows, b_vals = [], []

    # y_t ≤ x_t  →  y_t − x_t ≤ 0
    for t in range(n):
        row = np.zeros(m)
        row[n + t] = 1.0
        row[t]     = -1.0
        A_rows.append(row)
        b_vals.append(0.0)

    # Energy budget: Σ x_t · 0.5 ≤ e_budget
    row = np.zeros(m)
    row[:n] = PERIOD_HOURS
    A_rows.append(row)
    b_vals.append(e_budget)

    # Ramp constraints: x_t − x_{t−1} ≤ ramp  and  x_{t−1} − x_t ≤ ramp
    if ramp_mw < sc.fleet_mw and n >= 2:
        for t in range(1, n):
            r_up = np.zeros(m)
            r_up[t] = 1.0; r_up[t-1] = -1.0
            A_rows.append(r_up)
            b_vals.append(ramp_mw)

            r_dn = np.zeros(m)
            r_dn[t] = -1.0; r_dn[t-1] = 1.0
            A_rows.append(r_dn)
            b_vals.append(ramp_mw)

    res = linprog(c=c, A_ub=np.array(A_rows), b_ub=np.array(b_vals),
                  bounds=bounds, method="highs")

    if not res.success:
        # Fallback to greedy
        x = np.minimum(cur_eff, avail)
        return x, x.copy()

    return res.x[:n], res.x[n:]


def run_scenario(df: pd.DataFrame, sc: Scenario) -> Dict:
    """Run both greedy and optimised dispatch for one scenario."""
    # Greedy
    y_greedy     = greedy_dispatch(df, sc)
    greedy_gwh   = y_greedy.sum() * PERIOD_HOURS / 1000
    total_mwh    = df["curtailment_MWh"].sum()
    greedy_pct   = (greedy_gwh * 1000 / total_mwh) if total_mwh > 0 else np.nan

    # Optimised (per-day LP)
    y_opt_all = np.zeros(len(df))
    offset = 0
    example_pack = None
    daily_cur = df.groupby("date")["curtailment_MWh"].sum()
    peak_date = daily_cur.idxmax() if len(daily_cur) else None

    for d, day_df in df.groupby("date"):
        cur_eff = sc.deliverability * day_df["curtailment_MW"].to_numpy()
        avail   = sc.fleet_mw * (day_df["dc_headroom_pct"].to_numpy() / 100.0) * sc.flex_frac
        x, y    = optimize_one_day(cur_eff, avail, sc)

        y_opt_all[offset:offset+len(y)] = y
        offset += len(y)

        if d == peak_date:
            example_pack = (day_df.index, cur_eff, avail, x, y)

    opt_gwh = y_opt_all.sum() * PERIOD_HOURS / 1000
    opt_pct = (opt_gwh * 1000 / total_mwh) if total_mwh > 0 else np.nan

    return {
        "scenario":           sc.name,
        "fleet_mw":           sc.fleet_mw,
        "flex_frac":          sc.flex_frac,
        "deliverability":     sc.deliverability,
        "backlog_frac":       sc.backlog_frac,
        "ramp_frac":          sc.ramp_frac,
        "greedy_GWh":         greedy_gwh,
        "greedy_pct":         greedy_pct,
        "opt_GWh":            opt_gwh,
        "opt_pct":            opt_pct,
        "y_opt_MW":           y_opt_all,
        "y_greedy_MW":        y_greedy,
        "example":            example_pack,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def make_plots(df, results, summary, suffix=""):
    """Generate all optimisation charts."""
    tag = f" [{suffix.strip('_')}]" if suffix else ""

    # ── 4a. Greedy vs Optimised bar chart ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(summary))
    w = 0.35
    ax.bar(x - w/2, summary["greedy_GWh"], width=w,
           color=COLORS["blue_light"], edgecolor="white", label="Greedy (per-period min)")
    ax.bar(x + w/2, summary["opt_GWh"], width=w,
           color=COLORS["blue"], edgecolor="white", label="Optimised (LP with constraints)")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["scenario"], rotation=18, ha="right", fontsize=8)
    ax.set_ylabel("Absorbed Energy (GWh)")
    ax.set_title(f"Curtailment Absorption: Greedy vs Optimised{tag}")
    ax.legend()
    for i, (g, o) in enumerate(zip(summary["greedy_GWh"], summary["opt_GWh"])):
        ax.text(i + w/2, o + summary["opt_GWh"].max() * 0.01, f"{o:.0f}",
                ha="center", fontsize=8, fontweight="bold")
    save_fig(fig, os.path.join(CHART_DIR, f"04_opt_vs_greedy{suffix}.png"))

    # ── 4b. Constraint waterfall ──────────────────────────────────────────
    # Shows how each added constraint reduces achievable absorption
    fig, ax = plt.subplots(figsize=(12, 5))
    vals  = summary["opt_GWh"].values
    names = summary["scenario"].values
    base  = vals[0]  # ideal / unconstrained

    colors_wf = [COLORS["green"]] + [COLORS["red"]] * (len(vals) - 1)
    bars = ax.bar(range(len(vals)), vals, color=colors_wf, alpha=0.85, edgecolor="white")

    # Draw reduction annotations
    for i in range(1, len(vals)):
        delta = vals[i] - vals[i-1]
        if abs(delta) > 0.5:
            mid_y = (vals[i] + vals[i-1]) / 2
            ax.annotate(f"{delta:+.0f} GWh",
                        xy=(i, mid_y), fontsize=8, ha="center",
                        color=COLORS["red"], fontweight="bold")

    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Absorbed Energy (GWh)")
    ax.set_title(f"Constraint Waterfall: Impact of Operational Realism{tag}")
    for i, v in enumerate(vals):
        ax.text(i, v + base * 0.005, f"{v:.0f}", ha="center", fontsize=8, fontweight="bold")
    save_fig(fig, os.path.join(CHART_DIR, f"04_waterfall{suffix}.png"))

    # ── 4c. Daily totals (top scenario) ───────────────────────────────────
    top = results[0]  # first scenario (ideal) for max visibility
    ts_plot = df.copy()
    ts_plot["absorbed_MWh_opt"] = top["y_opt_MW"] * PERIOD_HOURS
    daily = ts_plot.groupby("date")[["curtailment_MWh", "absorbed_MWh_opt"]].sum() / 1000

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(pd.to_datetime(daily.index), daily["curtailment_MWh"],
            label="Daily curtailed (GWh)", color=COLORS["blue"], alpha=0.7)
    ax.plot(pd.to_datetime(daily.index), daily["absorbed_MWh_opt"],
            label="Daily absorbed (GWh)", color=COLORS["green"])
    ax.set_title(f"Daily Curtailment vs Optimised Absorption — {top['scenario']}{tag}")
    ax.set_ylabel("GWh / day")
    ax.legend()
    save_fig(fig, os.path.join(CHART_DIR, f"04_daily_totals{suffix}.png"))

    # ── 4d. Example week around peak ──────────────────────────────────────
    peak_day = pd.to_datetime(daily["curtailment_MWh"].idxmax())
    wk_start = peak_day - pd.Timedelta(days=3)
    wk_end   = peak_day + pd.Timedelta(days=4)
    week = ts_plot.loc[(ts_plot.index >= wk_start) & (ts_plot.index < wk_end)]

    if len(week):
        sc = SCENARIOS[0]
        avail_wk = sc.fleet_mw * (week["dc_headroom_pct"].to_numpy() / 100.0) * sc.flex_frac

        fig, ax = plt.subplots(figsize=(13, 4))
        ax.plot(week.index, week["curtailment_MW"], label="Curtailment (MW)",
                color=COLORS["blue"])
        ax.plot(week.index, avail_wk, label="Available flex (MW)",
                color=COLORS["amber"], alpha=0.7)
        ax.plot(week.index, week["absorbed_MWh_opt"] / PERIOD_HOURS,
                label="Absorbed (MW)", color=COLORS["green"])
        ax.set_title(f"Example Week Around Peak ({peak_day.date()}){tag}")
        ax.set_ylabel("MW")
        ax.legend()
        save_fig(fig, os.path.join(CHART_DIR, f"04_example_week{suffix}.png"))

    # ── 4e. Example day LP dispatch ───────────────────────────────────────
    # Use the most constrained scenario for visual interest
    constrained = results[-1]
    if constrained["example"] is not None:
        idx, cur_eff, avail, x, y = constrained["example"]

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(idx, cur_eff, label="Effective curtailment (MW)",
                color=COLORS["blue"])
        ax.plot(idx, avail, label="Available flex (MW)",
                color=COLORS["amber"], alpha=0.7)
        ax.step(idx, x, where="post", label="Dispatched load x (MW)",
                color=COLORS["grey"], linestyle="--")
        ax.step(idx, y, where="post", label="Absorbed y (MW)",
                color=COLORS["green"], linewidth=2)
        ax.set_title(f"LP Dispatch — Peak Day | {constrained['scenario']}{tag}")
        ax.set_ylabel("MW")
        ax.set_xlabel("Time")
        plt.xticks(rotation=30)
        ax.legend(fontsize=8)
        save_fig(fig, os.path.join(CHART_DIR, f"04_example_day_dispatch{suffix}.png"))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_analysis(df: pd.DataFrame, scenarios: List[Scenario], suffix: str = ""):
    """Full pipeline: run all scenarios, save results, generate plots."""
    total_gwh = df["curtailment_MWh"].sum() / 1000
    tag = f" [{suffix.strip('_')}]" if suffix else ""
    print(f"\n  Total curtailment: {total_gwh:,.1f} GWh{tag}")

    results = []
    for sc in scenarios:
        r = run_scenario(df, sc)
        results.append(r)
        print(f"    {sc.name}: greedy={r['greedy_GWh']:.1f} GWh, "
              f"opt={r['opt_GWh']:.1f} GWh ({r['opt_pct']*100:.1f}%)")

    summary = pd.DataFrame([{k: v for k, v in r.items()
                              if k not in ("y_opt_MW", "y_greedy_MW", "example")}
                             for r in results])
    summary.to_csv(os.path.join(CSV_DIR, f"04_summary_results{suffix}.csv"), index=False)

    make_plots(df, results, summary, suffix=suffix)
    return summary


def main():
    print("=" * 70)
    print("04 — OPTIMISATION: CONSTRAINED DC CURTAILMENT ABSORPTION")
    print("=" * 70)

    df = load_timeseries()
    print(f"  Time series: {df.index.min()} → {df.index.max()} ({len(df):,} periods)")

    # General analysis
    summary = run_analysis(df, SCENARIOS, suffix="")

    # Optional farm group analysis
    if FARM_GROUP is not None:
        group_name, farm_names = FARM_GROUP
        print(f"\n  Farm group analysis: {group_name} ({len(farm_names)} farms)")
        try:
            farm_ts = build_farm_ts(farm_names, df)
            run_analysis(farm_ts, SCENARIOS, suffix=f"_{group_name}")
        except Exception as e:
            print(f"  ERROR in farm analysis: {e}")

    # Print summary table
    print(f"\n{'='*70}")
    print("SUMMARY — OPTIMISED ABSORPTION (1 GW DC fleet)")
    print(f"{'='*70}")
    print(summary[["scenario", "opt_GWh", "opt_pct"]].to_string(index=False))

    print(f"\n  ✓ Results saved to {CSV_DIR}/04_*.csv")
    print(f"  ✓ Charts saved to  {CHART_DIR}/04_*.png")
    print("  Done.\n")


if __name__ == "__main__":
    main()
