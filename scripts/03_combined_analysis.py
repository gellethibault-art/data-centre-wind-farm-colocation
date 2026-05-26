"""
03_combined_analysis.py — Curtailment ↔ Data Centre Absorption Matching
Group 10: Wind Curtailment and Data Centres

Core question: for each half-hour settlement period, how much curtailed wind
energy could data centres realistically absorb?

This script performs TWO levels of analysis:
  PART A – SCENARIO MODELLING (hourly / seasonal profiles)
    Uses aggregated DC profiles to run fleet-size and flex-fraction scenarios.
    Quick, interpretable, good for the report's main results.

  PART B – HALF-HOURLY FLUCTUATION ANALYSIS (real time series)
    If UKPN half-hourly data is available, builds a continuous 30-min timeline
    with both curtailment and DC headroom, then computes period-by-period
    absorption.  Captures within-day volatility that averages hide.

Input:  data/processed/curtailment_per_period.csv    (from 01)
        data/processed/dc_hourly_profile_matched.csv (from 02)
        data/processed/dc_monthly_hourly_profile.csv (from 02, optional)
        data/raw/ukpn_dc_demand.csv                  (for Part B, optional)
Output: output/csv/absorption_summary.csv
        output/csv/sensitivity_flex_fraction.csv
        output/csv/matched_periods.csv
        output/csv/03_matched_timeseries.csv         (Part B)
        output/charts/03_*.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os, sys

# ── paths & style ─────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR   = os.path.join(ROOT, "data", "raw")
PROC_DIR  = os.path.join(ROOT, "data", "processed")
CHART_DIR = os.path.join(ROOT, "output", "charts")
CSV_DIR   = os.path.join(ROOT, "output", "csv")
for d in [CHART_DIR, CSV_DIR]:
    os.makedirs(d, exist_ok=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import apply_style, COLORS, SCENARIO_COLORS, save_fig, hour_labels
apply_style()

# ── config ────────────────────────────────────────────────────────────────────
START_DATE = "2024-04-01"
END_DATE   = "2026-04-01"  # exclusive

SCENARIOS = [
    {"name": "Small DC cluster (200 MW)",   "fleet_mw":  200, "flex_frac": 0.5},
    {"name": "Medium portfolio (500 MW)",   "fleet_mw":  500, "flex_frac": 0.5},
    {"name": "Large portfolio (1 GW)",      "fleet_mw": 1000, "flex_frac": 0.5},
    {"name": "Full UK flex (2 GW)",         "fleet_mw": 2000, "flex_frac": 0.5},
]

SENSITIVITY_FLEET_MW = 1000
FLEX_FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]

# Part B: real half-hourly analysis config
FLEET_MW       = 1000   # MW for the fluctuation analysis
FLEX_FRACTION  = 0.60   # fraction of headroom that is dispatchable
DELIVERABILITY = 1.00   # 1.0 = co-located, <1.0 = remote/constrained
CHUNK_SIZE     = 1_000_000  # for reading large UKPN CSV

# ══════════════════════════════════════════════════════════════════════════════
# PART A — SCENARIO MODELLING (profile-based matching)
# ══════════════════════════════════════════════════════════════════════════════

curt_path   = os.path.join(PROC_DIR, "curtailment_per_period.csv")
dc_path     = os.path.join(PROC_DIR, "dc_hourly_profile_matched.csv")
dc_mh_path  = os.path.join(PROC_DIR, "dc_monthly_hourly_profile.csv")

for path, script in [(curt_path, "01"), (dc_path, "02")]:
    if not os.path.exists(path):
        sys.exit(f"ERROR: {path} not found. Run {script}_*.py first.")

print("Loading processed data …")
sp = pd.read_csv(curt_path)
sp["Date"] = pd.to_datetime(sp["Date"])
sp["Month"]    = sp["Date"].dt.to_period("M")
sp["MonthInt"] = sp["Date"].dt.month

dc_profile = pd.read_csv(dc_path, index_col=0)
try:
    dc_profile.index = dc_profile.index.astype(int)
except Exception:
    pass

USE_SEASONAL = os.path.exists(dc_mh_path)
if USE_SEASONAL:
    dc_mh = pd.read_csv(dc_mh_path, index_col=0)
    try:
        dc_mh.index = dc_mh.index.astype(int)
    except Exception:
        pass
    dc_mh.columns = dc_mh.columns.astype(int)
    print(f"  ✓ Seasonal month×hour profile loaded")
else:
    print("  ℹ No seasonal profile — using hourly average only")

print(f"  Curtailment periods: {len(sp):,}")
print(f"  DC profile: mean util = {dc_profile['mean_util'].mean():.1f}%, "
      f"mean headroom = {dc_profile['flexibility_pct'].mean():.1f}%")

# ── map DC headroom onto each settlement period ──────────────────────────────
if USE_SEASONAL:
    util_lookup = (dc_mh.stack()
                        .rename("dc_util_pct")
                        .reset_index()
                        .rename(columns={"hour": "HourInt", "level_1": "MonthInt"}))
    sp = sp.merge(util_lookup, on=["HourInt", "MonthInt"], how="left")
    sp["dc_util_pct"] = sp["dc_util_pct"].fillna(sp["HourInt"].map(dc_profile["mean_util"]))
    profile_label = "seasonal (month × hour)"
else:
    sp["dc_util_pct"] = sp["HourInt"].map(dc_profile["mean_util"]).fillna(dc_profile["mean_util"].mean())
    profile_label = "hourly average"

sp["dc_headroom_pct"] = (100 - sp["dc_util_pct"]).clip(lower=0)
print(f"  Using {profile_label} matching | mean headroom: {sp['dc_headroom_pct'].mean():.1f}%")

# ── run scenarios ─────────────────────────────────────────────────────────────
total_curtailed_gwh = sp["total_curtailment_MWh"].sum() / 1000
summary_rows = []

print(f"\n{'='*60}")
print(f"SCENARIO RESULTS  (total curtailed: {total_curtailed_gwh:,.0f} GWh)")
print(f"{'='*60}")

for scen in SCENARIOS:
    fleet, flex, name = scen["fleet_mw"], scen["flex_frac"], scen["name"]
    avail_mw    = fleet * (sp["dc_headroom_pct"] / 100) * flex
    absorbed_mw = np.minimum(sp["curtailment_MW"].values, avail_mw.values)
    absorbed_mwh = absorbed_mw * 0.5

    sp[f"absorbed_{name}_MWh"] = absorbed_mwh
    sp[f"flex_avail_{name}_MW"] = avail_mw

    gwh  = absorbed_mwh.sum() / 1000
    pct  = gwh / total_curtailed_gwh * 100
    fully = (sp["curtailment_MW"].values <= avail_mw.values).sum()

    summary_rows.append({
        "Scenario": name, "Fleet_MW": fleet, "Flex_Fraction": flex,
        "Absorbed_GWh": round(gwh, 1), "Pct_of_Curtailment": round(pct, 1),
        "Periods_Fully_Absorbed": fully,
        "Pct_Fully_Absorbed": round(fully / len(sp) * 100, 1),
    })
    print(f"  {name}: {gwh:,.1f} GWh absorbed ({pct:.1f}%)")

summary = pd.DataFrame(summary_rows)
summary.to_csv(os.path.join(CSV_DIR, "absorption_summary.csv"), index=False)
sp.to_csv(os.path.join(CSV_DIR, "matched_periods.csv"), index=False)
print(f"  ✓ Saved absorption_summary.csv + matched_periods.csv")

# ── sensitivity analysis ──────────────────────────────────────────────────────
print(f"\n  Sensitivity ({SENSITIVITY_FLEET_MW} MW fleet, varying flex):")
sens_rows = []
for frac in FLEX_FRACTIONS:
    avail    = SENSITIVITY_FLEET_MW * (sp["dc_headroom_pct"] / 100) * frac
    absorbed = np.minimum(sp["curtailment_MW"].values, avail.values) * 0.5
    gwh = absorbed.sum() / 1000
    sens_rows.append({"flex_fraction": frac, "absorbed_gwh": gwh,
                      "pct": gwh / total_curtailed_gwh * 100})
    print(f"    flex={frac:.0%}: {gwh:,.0f} GWh ({gwh/total_curtailed_gwh*100:.1f}%)")
sens_df = pd.DataFrame(sens_rows)
sens_df.to_csv(os.path.join(CSV_DIR, "sensitivity_flex_fraction.csv"), index=False)


# ══════════════════════════════════════════════════════════════════════════════
# PART B — HALF-HOURLY FLUCTUATION ANALYSIS (real time series)
# ══════════════════════════════════════════════════════════════════════════════
# Builds a continuous 30-min timeline of curtailment AND DC utilisation,
# showing the real within-day volatility that hourly averages smooth out.

UKPN_PATH = os.path.join(RAW_DIR, "ukpn_dc_demand.csv")
HAS_UKPN  = os.path.exists(UKPN_PATH)

ts = None  # will be set if Part B runs

if HAS_UKPN:
    print(f"\n{'='*60}")
    print("PART B: HALF-HOURLY FLUCTUATION ANALYSIS")
    print(f"{'='*60}")

    # ── load curtailment into full half-hour index ────────────────────────
    sp_b = pd.read_csv(curt_path)
    if "Datetime" in sp_b.columns:
        sp_b["Datetime"] = pd.to_datetime(sp_b["Datetime"], errors="coerce")
    else:
        sp_b["Date"] = pd.to_datetime(sp_b["Date"], errors="coerce")
        sp_b["Datetime"] = sp_b["Date"] + pd.to_timedelta(
            (sp_b["Settlement_Period"].astype(int) - 1) * 30, unit="m")
    sp_b = sp_b.dropna(subset=["Datetime"])
    sp_b = sp_b[(sp_b["Datetime"] >= pd.Timestamp(START_DATE)) &
                (sp_b["Datetime"] <  pd.Timestamp(END_DATE))]

    if "curtailment_MW" in sp_b.columns:
        curt_mw = sp_b.groupby("Datetime")["curtailment_MW"].sum()
    else:
        curt_mw = sp_b.groupby("Datetime")["total_curtailment_MWh"].sum() / 0.5

    full_idx = pd.date_range(START_DATE, END_DATE, freq="30min", inclusive="left")
    curt_mw  = curt_mw.reindex(full_idx, fill_value=0.0)
    curt_mwh = curt_mw * 0.5

    # ── load UKPN half-hourly utilisation (chunked for memory) ────────────
    print("  Building half-hour DC utilisation time series (chunked) …")
    start_utc = pd.Timestamp(START_DATE).tz_localize("UTC")
    end_utc   = pd.Timestamp(END_DATE).tz_localize("UTC")
    usecols   = ["utc_timestamp", "anonymised_data_centre_name", "hh_utilisation_ratio"]
    acc = None

    for i, chunk in enumerate(pd.read_csv(UKPN_PATH, usecols=usecols, chunksize=CHUNK_SIZE)):
        t = pd.to_datetime(chunk["utc_timestamp"], errors="coerce", utc=True)
        m = (t >= start_utc) & (t < end_utc)
        if not m.any():
            continue
        sub = chunk.loc[m, ["anonymised_data_centre_name", "hh_utilisation_ratio"]].copy()
        sub["t"] = t.loc[m].values
        sub = sub.groupby(["t", "anonymised_data_centre_name"], as_index=False)["hh_utilisation_ratio"].mean()
        sub["sq"] = sub["hh_utilisation_ratio"].astype("float64") ** 2
        g = sub.groupby("t").agg(
            n=("hh_utilisation_ratio", "count"),
            s=("hh_utilisation_ratio", "sum"),
            ss=("sq", "sum"),
        )
        acc = g if acc is None else acc.add(g, fill_value=0)
        if (i + 1) % 5 == 0:
            print(f"    processed chunks: {i+1}")

    if acc is None or acc.empty:
        print("  WARNING: No UKPN rows found in date window. Skipping Part B.")
        HAS_UKPN = False  # disable Part B charts
    else:
        dc_mean = acc["s"] / acc["n"]
        dc_var  = (acc["ss"] / acc["n"]) - (dc_mean ** 2)
        dc_std  = np.sqrt(np.maximum(dc_var, 0.0))

        tmp = pd.DataFrame({"dc_mean": dc_mean, "dc_std": dc_std})
        tmp.index = pd.to_datetime(tmp.index, utc=True, errors="coerce")
        tmp = tmp.loc[~tmp.index.isna()]
        tmp.index = tmp.index.tz_convert(None)

        dc = pd.DataFrame({
            "dc_util_pct": tmp["dc_mean"] * 100,
            "dc_util_std_pct": tmp["dc_std"] * 100,
        }, index=tmp.index)
        dc = dc.reindex(full_idx).interpolate(limit=4).ffill().bfill()
        dc["dc_headroom_pct"] = 100 - dc["dc_util_pct"]

        # ── compute flexible capacity & absorption per period ─────────────
        flex_mw     = FLEET_MW * (dc["dc_headroom_pct"] / 100.0) * FLEX_FRACTION * DELIVERABILITY
        flex_mw     = flex_mw.clip(lower=0.0)
        absorbed_mw = np.minimum(curt_mw.values, flex_mw.values)

        ts = pd.DataFrame({
            "curtailment_MW":   curt_mw.values,
            "curtailment_MWh":  curt_mwh.values,
            "dc_util_pct":      dc["dc_util_pct"].values,
            "dc_headroom_pct":  dc["dc_headroom_pct"].values,
            "flex_MW":          flex_mw.values,
            "absorbed_MW":      absorbed_mw,
            "absorbed_MWh":     absorbed_mw * 0.5,
        }, index=full_idx)
        ts.index.name = "Datetime"
        ts.to_csv(os.path.join(CSV_DIR, "03_matched_timeseries.csv"))

        total_curt_b = ts["curtailment_MWh"].sum()
        total_abs_b  = ts["absorbed_MWh"].sum()
        pct_b = (total_abs_b / total_curt_b * 100) if total_curt_b > 0 else 0

        print(f"\n  Fleet: {FLEET_MW} MW | Flex: {FLEX_FRACTION:.0%} | Deliverability: {DELIVERABILITY}")
        print(f"  Total curtailed: {total_curt_b/1e6:.2f} TWh")
        print(f"  Total absorbed:  {total_abs_b/1e6:.2f} TWh ({pct_b:.1f}%)")
        print(f"  ✓ Saved 03_matched_timeseries.csv")
else:
    print("\n  ℹ UKPN raw data not found — Part B (fluctuation analysis) skipped.")
    print("    Place ukpn_dc_demand.csv in data/raw/ to enable it.")


# ══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════════════

print(f"\nGenerating charts …")

# ── 3a. Scenario comparison (absorbed GWh + % of curtailment) ─────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.bar(range(len(summary)), summary["Absorbed_GWh"],
        color=SCENARIO_COLORS[:len(summary)], alpha=0.85)
ax1.axhline(y=total_curtailed_gwh, color="black", linestyle=":",
            linewidth=1, label=f"Total curtailed ({total_curtailed_gwh:,.0f} GWh)")
ax1.set_xticks(range(len(summary)))
ax1.set_xticklabels(summary["Scenario"], rotation=20, ha="right", fontsize=8)
ax1.set_ylabel("Energy Absorbed (GWh)")
ax1.set_title("Curtailment Absorbed by Data Centres")
ax1.legend(fontsize=8)
for i, v in enumerate(summary["Absorbed_GWh"]):
    ax1.text(i, v + total_curtailed_gwh * 0.01, f"{v:,.0f}",
             ha="center", fontsize=9, fontweight="bold")

ax2.bar(range(len(summary)), summary["Pct_of_Curtailment"],
        color=SCENARIO_COLORS[:len(summary)], alpha=0.85)
ax2.set_xticks(range(len(summary)))
ax2.set_xticklabels(summary["Scenario"], rotation=20, ha="right", fontsize=8)
ax2.set_ylabel("% of Curtailment Absorbed")
ax2.set_title("Fraction Mitigated")
ax2.set_ylim(0, max(summary["Pct_of_Curtailment"]) * 1.3)
for i, v in enumerate(summary["Pct_of_Curtailment"]):
    ax2.text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")

save_fig(fig, os.path.join(CHART_DIR, "03_scenario_comparison.png"))


# ── 3b. Hourly overlay: curtailment vs flex capacity ─────────────────────────

hourly_curt = sp.groupby("HourInt")["curtailment_MW"].mean()

fig, ax = plt.subplots(figsize=(13, 6))
ax.bar(range(24), hourly_curt.reindex(range(24), fill_value=0),
       color=COLORS["blue_pale"], width=0.85, label="Mean curtailment (MW)")
for i, scen in enumerate(SCENARIOS):
    hourly_flex = sp.groupby("HourInt")[f"flex_avail_{scen['name']}_MW"].mean()
    ax.plot(range(24), hourly_flex.reindex(range(24), fill_value=0),
            color=SCENARIO_COLORS[i], linewidth=2.5, marker="o", markersize=4,
            label=scen["name"])
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Power (MW)")
ax.set_title("Mean Hourly Curtailment vs DC Flexible Capacity")
hour_labels(ax)
ax.legend(fontsize=8, loc="upper right")
save_fig(fig, os.path.join(CHART_DIR, "03_hourly_overlay.png"))


# ── 3c. Monthly absorption ───────────────────────────────────────────────────

month_order = pd.period_range("2024-04", "2026-03", freq="M")
monthly_curt = sp.groupby("Month")["total_curtailment_MWh"].sum().reindex(month_order) / 1000

fig, ax = plt.subplots(figsize=(14, 5))
x = np.arange(len(month_order))
ax.bar(x, monthly_curt.values, color=COLORS["blue_pale"], width=0.85, label="Total curtailment")
width = 0.18
offsets = np.linspace(-0.28, 0.28, len(SCENARIOS))
for i, scen in enumerate(SCENARIOS):
    col = f"absorbed_{scen['name']}_MWh"
    monthly_abs = sp.groupby("Month")[col].sum().reindex(month_order) / 1000
    ax.bar(x + offsets[i], monthly_abs.values, width=width,
           color=SCENARIO_COLORS[i], alpha=0.85, label=scen["name"])
ax.set_xticks(x)
ax.set_xticklabels([m.strftime("%b\n%Y") if m.month in [1, 4, 7, 10] else m.strftime("%b")
                     for m in month_order], fontsize=8)
ax.set_ylabel("Energy (GWh)")
ax.set_title("Monthly Curtailment & DC Absorption by Scenario")
ax.legend(fontsize=7, ncol=3)
save_fig(fig, os.path.join(CHART_DIR, "03_monthly_absorption.png"))


# ── 3d. Sensitivity: flex fraction ───────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(sens_df["flex_fraction"] * 100, sens_df["absorbed_gwh"],
        color=COLORS["blue"], linewidth=2.5, marker="o", markersize=8)
ax.fill_between(sens_df["flex_fraction"] * 100, 0, sens_df["absorbed_gwh"],
                alpha=0.1, color=COLORS["blue"])
for _, row in sens_df.iterrows():
    ax.annotate(f"{row['absorbed_gwh']:,.0f} GWh\n({row['pct']:.1f}%)",
                xy=(row["flex_fraction"] * 100, row["absorbed_gwh"]),
                textcoords="offset points", xytext=(0, 12), ha="center", fontsize=9)
ax.set_xlabel("Flexibility Fraction (% of headroom that is shiftable)")
ax.set_ylabel("Total Absorbed (GWh)")
ax.set_title(f"Sensitivity — {SENSITIVITY_FLEET_MW} MW DC Fleet, Varying Flexibility")
save_fig(fig, os.path.join(CHART_DIR, "03_sensitivity_flex.png"))


# ── 3e. KEY INSIGHT: temporal alignment ──────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
ax1.bar(range(24), hourly_curt.reindex(range(24), fill_value=0) / 1000,
        color=COLORS["blue"], alpha=0.85)
ax1.set_ylabel("Mean Curtailment (GW)")
ax1.set_title("The Opportunity: Wind Curtailment Peaks When DCs Have Most Headroom")

dc_headroom_hourly = sp.groupby("HourInt")["dc_headroom_pct"].mean()
ax2.bar(range(24), dc_headroom_hourly.reindex(range(24), fill_value=0),
        color=COLORS["green"], alpha=0.85)
ax2.set_ylabel("DC Headroom (% of capacity)")
ax2.set_xlabel("Hour of Day")
hour_labels(ax2)
save_fig(fig, os.path.join(CHART_DIR, "03_alignment_insight.png"))


# ── 3f. Dual heatmap: curtailment vs headroom ────────────────────────────────

if USE_SEASONAL:
    month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                   7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    curt_mh_data = sp.groupby(["MonthInt", "HourInt"])["curtailment_MW"].mean().reset_index()
    curt_pivot_  = curt_mh_data.pivot(index="HourInt", columns="MonthInt", values="curtailment_MW")
    months_avail = sorted(set(curt_pivot_.columns) & set(range(1, 13)))

    im1 = ax1.imshow(curt_pivot_[months_avail].values / 1000, aspect="auto",
                      cmap="YlOrRd", interpolation="nearest")
    ax1.set_yticks(range(24))
    ax1.set_yticklabels([f"{h:02d}" for h in range(24)])
    ax1.set_xticks(range(len(months_avail)))
    ax1.set_xticklabels([month_names.get(m, str(m)) for m in months_avail], rotation=45, ha="right")
    ax1.set_ylabel("Hour")
    ax1.set_title("Mean Curtailment (GW)")
    plt.colorbar(im1, ax=ax1, shrink=0.7, label="GW")

    headroom_mh_ = 100 - dc_mh
    mh_cols = sorted(set(headroom_mh_.columns) & set(months_avail))
    im2 = ax2.imshow(headroom_mh_[mh_cols].values, aspect="auto",
                      cmap="Greens", interpolation="nearest")
    ax2.set_yticks(range(24))
    ax2.set_yticklabels([f"{h:02d}" for h in range(24)])
    ax2.set_xticks(range(len(mh_cols)))
    ax2.set_xticklabels([month_names.get(m, str(m)) for m in mh_cols], rotation=45, ha="right")
    ax2.set_ylabel("Hour")
    ax2.set_title("DC Headroom (% of capacity)")
    plt.colorbar(im2, ax=ax2, shrink=0.7, label="%")

    fig.suptitle("Curtailment vs Data Centre Headroom — Month × Hour", fontsize=14, y=1.01)
    save_fig(fig, os.path.join(CHART_DIR, "03_dual_heatmap.png"))


# ── PART B CHARTS (half-hourly fluctuation analysis) ─────────────────────────

if ts is not None:
    print("  Generating Part B fluctuation charts …")

    # 3g. Daily totals (shows day-to-day volatility)
    daily_b = ts.resample("D").sum(numeric_only=True)
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(daily_b.index, daily_b["curtailment_MWh"] / 1000,
            label="Daily curtailed (GWh)", color=COLORS["blue"])
    ax.plot(daily_b.index, daily_b["absorbed_MWh"] / 1000,
            label="Daily absorbed (GWh)", color=COLORS["green"])
    ax.set_title(f"Daily Curtailment vs Absorbed Energy ({FLEET_MW} MW fleet)")
    ax.set_ylabel("Energy (GWh/day)")
    ax.legend()
    save_fig(fig, os.path.join(CHART_DIR, "03_fluct_daily_totals.png"))

    # 3h. Hour-of-day quantile bands (within-day shape + spread)
    ts_h = ts.copy()
    ts_h["Hour"] = ts_h.index.hour

    curt_q = ts_h.groupby("Hour")["curtailment_MW"].quantile([0.1, 0.5, 0.9]).unstack()
    curt_q.columns = ["p10", "p50", "p90"]
    flex_q = ts_h.groupby("Hour")["flex_MW"].quantile([0.1, 0.5, 0.9]).unstack()
    flex_q.columns = ["p10", "p50", "p90"]

    fig, ax = plt.subplots(figsize=(12, 5))
    h = curt_q.index.to_numpy(dtype=float)
    ax.fill_between(h, curt_q["p10"].astype(float), curt_q["p90"].astype(float),
                    alpha=0.2, color=COLORS["blue"], label="Curtailment MW (p10–p90)")
    ax.plot(h, curt_q["p50"].astype(float), color=COLORS["blue"], label="Curtailment MW (median)")
    ax.fill_between(h, flex_q["p10"].astype(float), flex_q["p90"].astype(float),
                    alpha=0.2, color=COLORS["green"], label="Flex MW (p10–p90)")
    ax.plot(h, flex_q["p50"].astype(float), color=COLORS["green"], label="Flex MW (median)")
    ax.set_title("Hourly Fluctuations: Distribution Across Year (Half-hour resolution)")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Power (MW)")
    hour_labels(ax)
    ax.legend(ncol=2, fontsize=8)
    save_fig(fig, os.path.join(CHART_DIR, "03_fluct_hourly_quantiles.png"))

    # 3i. Example week around peak curtailment day
    max_day = daily_b["curtailment_MWh"].idxmax()
    wk_start = max_day - pd.Timedelta(days=3)
    wk_end   = max_day + pd.Timedelta(days=4)
    w = ts[(ts.index >= wk_start) & (ts.index < wk_end)]

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(w.index, w["curtailment_MW"], label="Curtailment MW", color=COLORS["blue"])
    ax.plot(w.index, w["flex_MW"], label="Available Flex MW", color=COLORS["amber"])
    ax.plot(w.index, w["absorbed_MW"], label="Absorbed MW", color=COLORS["green"])
    ax.set_title(f"Example Week Around Peak Day ({max_day.date()})")
    ax.set_ylabel("MW")
    ax.legend()
    save_fig(fig, os.path.join(CHART_DIR, "03_fluct_week_example.png"))

    # 3j. Example single day dispatch (step plot)
    d0 = pd.Timestamp(max_day.date())
    d1 = d0 + pd.Timedelta(days=1)
    wd = ts[(ts.index >= d0) & (ts.index < d1)]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.step(wd.index, wd["curtailment_MW"], where="mid",
            label="Curtailment MW", color=COLORS["blue"])
    ax.step(wd.index, wd["flex_MW"], where="mid",
            label="Flex MW", color=COLORS["amber"])
    ax.step(wd.index, wd["absorbed_MW"], where="mid",
            label="Absorbed MW", color=COLORS["green"])
    ax.set_title(f"Example Day — {d0.date()} (Half-hour dispatch)")
    ax.set_ylabel("MW")
    ax.legend(fontsize=9, ncol=3)
    save_fig(fig, os.path.join(CHART_DIR, "03_fluct_example_day.png"))


# ── summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("KEY FINDINGS")
print(f"{'='*60}")
print(f"  1. {total_curtailed_gwh:,.0f} GWh curtailed across {len(sp):,} half-hour periods.")
print(f"  2. Curtailment peaks overnight when DC headroom is greatest — key opportunity.")
print(f"  3. Absorption by scenario:")
for _, r in summary.iterrows():
    print(f"     {r['Scenario']}: {r['Absorbed_GWh']:,.0f} GWh ({r['Pct_of_Curtailment']}%)")
print(f"  4. Profile type: {profile_label}")
print(f"\n  ✓ All outputs in {CHART_DIR}/03_*.png and {CSV_DIR}/")
print("  Done.\n")
