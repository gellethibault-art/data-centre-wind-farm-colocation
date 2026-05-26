"""
02_dc_demand_analysis.py — Data Centre Demand Profile Analysis
Group 10: Wind Curtailment and Data Centres

Analyses UKPN data centre demand profiles to understand:
  - Hourly utilisation patterns (when headroom is highest)
  - Differences by DC type (Enterprise vs Co-located) and voltage level
  - Seasonal variation (month × hour)
  - Date alignment with the BOA curtailment window (Apr 2024 – Mar 2026)

NOTE: UKPN reports utilisation ratios (0–1), NOT absolute MW.
      MW scaling uses assumed fleet capacities applied in scripts 03/04.

Input:  data/raw/ukpn_dc_demand.csv
Output: data/processed/dc_hourly_profile.csv
        data/processed/dc_hourly_profile_matched.csv
        data/processed/dc_monthly_hourly_profile.csv
        data/processed/dc_type_profiles.csv
        data/processed/dc_voltage_profiles.csv
        output/charts/02_*.png
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
for d in [PROC_DIR, CHART_DIR]:
    os.makedirs(d, exist_ok=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import apply_style, COLORS, save_fig, hour_labels
apply_style()

UKPN_PATH = os.path.join(RAW_DIR, "ukpn_dc_demand.csv")
USE_REAL_DATA = os.path.exists(UKPN_PATH)

BOA_START = pd.Timestamp("2024-04-01")
BOA_END   = pd.Timestamp("2026-03-31 23:59:59")

# ══════════════════════════════════════════════════════════════════════════════
# PATH A: REAL UKPN DATA
# ══════════════════════════════════════════════════════════════════════════════

if USE_REAL_DATA:
    print("Loading REAL UKPN data centre demand profiles …")

    # Opendatasoft exports may use semicolons — auto-detect
    try:
        dc_raw = pd.read_csv(UKPN_PATH, sep=";", low_memory=False)
        if len(dc_raw.columns) <= 2:
            dc_raw = pd.read_csv(UKPN_PATH, sep=",", low_memory=False)
    except Exception:
        dc_raw = pd.read_csv(UKPN_PATH, sep=",", low_memory=False)

    print(f"  Columns: {dc_raw.columns.tolist()}")
    print(f"  Shape:   {dc_raw.shape}")

    # ── standardise column names ──────────────────────────────────────────
    col_map = {}
    for c in dc_raw.columns:
        cl = c.lower().strip()
        if "voltage" in cl:
            col_map[c] = "voltage_level"
        elif "anonymised" in cl or "data_centre_name" in cl:
            col_map[c] = "site_id"
        elif cl in ("dc_type", "type"):
            col_map[c] = "dc_type"
        elif "local_timestamp" in cl:
            col_map[c] = "local_timestamp"
        elif "utc_timestamp" in cl:
            col_map[c] = "utc_timestamp"
        elif "utilisation" in cl or "ratio" in cl:
            col_map[c] = "hh_utilisation_ratio"
    dc_raw.rename(columns=col_map, inplace=True)

    # ── parse timestamps ──────────────────────────────────────────────────
    if "local_timestamp" in dc_raw.columns:
        dc_raw["datetime"] = (pd.to_datetime(dc_raw["local_timestamp"], utc=True, errors="coerce")
                                .dt.tz_convert("Europe/London").dt.tz_localize(None))
    elif "utc_timestamp" in dc_raw.columns:
        dc_raw["datetime"] = (pd.to_datetime(dc_raw["utc_timestamp"], utc=True, errors="coerce")
                                .dt.tz_convert("Europe/London").dt.tz_localize(None))
    else:
        sys.exit("ERROR: No timestamp column found in UKPN data.")
    dc_raw = dc_raw.dropna(subset=["datetime"])

    # ── parse utilisation ratio ───────────────────────────────────────────
    dc_raw["hh_utilisation_ratio"] = pd.to_numeric(dc_raw["hh_utilisation_ratio"], errors="coerce")
    dc_raw = dc_raw.dropna(subset=["hh_utilisation_ratio"])
    p99 = dc_raw["hh_utilisation_ratio"].quantile(0.99)
    if p99 <= 1.5:
        dc_raw["utilisation_pct"] = dc_raw["hh_utilisation_ratio"] * 100
        print("  Utilisation scale: RATIO (0–1) → converted to %")
    else:
        dc_raw["utilisation_pct"] = dc_raw["hh_utilisation_ratio"]
        print("  Utilisation scale: already PERCENTAGE")
    dc_raw["utilisation_pct"] = dc_raw["utilisation_pct"].clip(0, 150)

    # ── time features ─────────────────────────────────────────────────────
    dc_raw["hour"]  = dc_raw["datetime"].dt.hour
    dc_raw["month"] = dc_raw["datetime"].dt.month
    dc_raw["date"]  = dc_raw["datetime"].dt.date
    dc_raw["year"]  = dc_raw["datetime"].dt.year

    vl_map = {"Low Voltage Import": "LV", "High Voltage Import": "HV",
              "Extra-High Voltage Import": "EHV", "132kV Import": "132kV"}
    if "voltage_level" in dc_raw.columns:
        dc_raw["voltage_short"] = dc_raw["voltage_level"].map(vl_map).fillna(dc_raw["voltage_level"])
    else:
        dc_raw["voltage_short"] = "Unknown"
    if "dc_type" not in dc_raw.columns:
        dc_raw["dc_type"] = "Unknown"

    n_sites = dc_raw["site_id"].nunique() if "site_id" in dc_raw.columns else "N/A"
    print(f"\n  Full dataset: {len(dc_raw):,} records")
    print(f"  Date range:   {dc_raw['datetime'].min()} → {dc_raw['datetime'].max()}")
    print(f"  Unique sites: {n_sites}")
    print(f"  Voltage levels: {dc_raw['voltage_short'].value_counts().to_dict()}")
    print(f"  DC types:       {dc_raw['dc_type'].value_counts().to_dict()}")
    print(f"  Mean util:      {dc_raw['utilisation_pct'].mean():.1f}%")

    # ── date alignment ────────────────────────────────────────────────────
    overlap = dc_raw[(dc_raw["datetime"] >= BOA_START) & (dc_raw["datetime"] <= BOA_END)].copy()
    print(f"\n  OVERLAP with BOA (Apr 2024 – Mar 2026): {len(overlap):,} records "
          f"({len(overlap)/len(dc_raw)*100:.0f}% of dataset)")
    if len(overlap) == 0:
        print("  WARNING: No overlap! Using full dataset (DC patterns are yearly-consistent).")
        overlap = dc_raw.copy()

    dc = dc_raw  # full dataset for general profiles

    # ── build profiles (equal weight per site to avoid bias) ──────────────
    # Profile 1: all data
    site_hour     = dc.groupby(["site_id", "hour"])["utilisation_pct"].mean().reset_index()
    hourly_all    = site_hour.groupby("hour")["utilisation_pct"].agg(["mean", "std", "min", "max"])
    hourly_all.columns = ["mean_util", "std_util", "min_util", "max_util"]
    hourly_all["flexibility_pct"] = 100 - hourly_all["mean_util"]

    # Profile 2: overlap period only
    site_hour_ov  = overlap.groupby(["site_id", "hour"])["utilisation_pct"].mean().reset_index()
    hourly_match  = site_hour_ov.groupby("hour")["utilisation_pct"].agg(["mean", "std", "min", "max"])
    hourly_match.columns = ["mean_util", "std_util", "min_util", "max_util"]
    hourly_match["flexibility_pct"] = 100 - hourly_match["mean_util"]

    # Profile 3: month × hour (overlap)
    mh = overlap.groupby(["month", "hour"])["utilisation_pct"].mean().reset_index()
    monthly_hourly_pivot = mh.pivot(index="hour", columns="month", values="utilisation_pct")

    # Profile 4: by DC type
    type_profiles = dc.groupby(["hour", "dc_type"])["utilisation_pct"].mean().unstack()
    type_profiles = type_profiles.reindex(range(24))

    # Profile 5: by voltage level
    voltage_profiles = dc.groupby(["hour", "voltage_short"])["utilisation_pct"].mean().unstack()
    voltage_profiles = voltage_profiles.reindex(range(24))

    data_label = "UKPN Real Data"

    # ── save ──────────────────────────────────────────────────────────────
    hourly_all.to_csv(os.path.join(PROC_DIR, "dc_hourly_profile.csv"))
    hourly_match.to_csv(os.path.join(PROC_DIR, "dc_hourly_profile_matched.csv"))
    monthly_hourly_pivot.to_csv(os.path.join(PROC_DIR, "dc_monthly_hourly_profile.csv"))
    type_profiles.to_csv(os.path.join(PROC_DIR, "dc_type_profiles.csv"))
    voltage_profiles.to_csv(os.path.join(PROC_DIR, "dc_voltage_profiles.csv"))
    print("  ✓ Saved 5 processed profile files")

# ══════════════════════════════════════════════════════════════════════════════
# PATH B: SYNTHETIC PROFILES (fallback if UKPN not available)
# ══════════════════════════════════════════════════════════════════════════════
else:
    print("=" * 60)
    print("UKPN data not found — generating SYNTHETIC DC profiles.")
    print("Place real data as: data/raw/ukpn_dc_demand.csv")
    print("=" * 60)

    np.random.seed(42)
    hours = np.arange(24)
    enterprise_base = 78 + 5 * np.sin(2 * np.pi * (hours - 14) / 24)
    enterprise_base = np.clip(enterprise_base, 70, 88)
    colocated_base  = 62 + 12 * np.sin(2 * np.pi * (hours - 14) / 24)
    colocated_base  = np.clip(colocated_base, 50, 78)

    records = []
    for m in range(1, 13):
        seasonal = 1.0 + 0.03 * np.sin(2 * np.pi * (m - 7) / 12)
        for h in hours:
            for dc_type, base in [("Enterprise", enterprise_base), ("Co-located", colocated_base)]:
                records.append({"month": m, "hour": h, "dc_type": dc_type,
                                "utilisation_pct": base[h] * seasonal})
    dc = pd.DataFrame(records)

    hourly_all = dc.groupby("hour")["utilisation_pct"].agg(["mean", "std", "min", "max"])
    hourly_all.columns = ["mean_util", "std_util", "min_util", "max_util"]
    hourly_all["flexibility_pct"] = 100 - hourly_all["mean_util"]
    hourly_match = hourly_all.copy()

    type_profiles = dc.groupby(["hour", "dc_type"])["utilisation_pct"].mean().unstack()
    voltage_profiles = None
    monthly_hourly_pivot = None

    hourly_all.to_csv(os.path.join(PROC_DIR, "dc_hourly_profile.csv"))
    hourly_match.to_csv(os.path.join(PROC_DIR, "dc_hourly_profile_matched.csv"))
    type_profiles.to_csv(os.path.join(PROC_DIR, "dc_type_profiles.csv"))
    data_label = "Synthetic (Literature-Based)"


# ══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════════════

profile = hourly_match

# ── 2a. Hourly utilisation with headroom band ─────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 5))
ax.fill_between(profile.index, 100, profile["mean_util"],
                alpha=0.3, color=COLORS["green"], label="Available headroom")
ax.fill_between(profile.index, 0, profile["mean_util"],
                alpha=0.3, color=COLORS["blue"], label="Mean utilisation")
ax.plot(profile.index, profile["mean_util"],
        color=COLORS["blue"], linewidth=2.5, marker="o", markersize=4)
if profile["std_util"].notna().all():
    ax.fill_between(profile.index,
                    (profile["mean_util"] - profile["std_util"]).clip(lower=0),
                    (profile["mean_util"] + profile["std_util"]).clip(upper=100),
                    alpha=0.12, color=COLORS["blue"], label="±1 std dev")
ax.axhline(y=100, color="black", linestyle="-", linewidth=0.5)
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Utilisation (%)")
ax.set_title(f"Data Centre Hourly Load Profile — {data_label}\n"
             f"Green = headroom available to absorb curtailed wind")
hour_labels(ax)
ax.set_ylim(0, 105)
ax.legend(loc="lower right")
save_fig(fig, os.path.join(CHART_DIR, "02_dc_hourly_profile.png"))

# ── 2b. Enterprise vs Co-located ─────────────────────────────────────────────

if type_profiles is not None and len(type_profiles.columns) > 1:
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, col in enumerate(type_profiles.columns):
        ax.plot(type_profiles.index, type_profiles[col],
                linewidth=2.5, marker="o", markersize=4, label=col,
                color=[COLORS["blue"], COLORS["amber"]][i % 2])
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Mean Utilisation (%)")
    ax.set_title(f"Enterprise vs Co-located Data Centres — {data_label}")
    hour_labels(ax)
    ax.set_ylim(0, 105)
    ax.legend()
    save_fig(fig, os.path.join(CHART_DIR, "02_dc_type_comparison.png"))

# ── 2c. By voltage level ─────────────────────────────────────────────────────

if voltage_profiles is not None and len(voltage_profiles.columns) > 1:
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, col in enumerate(voltage_profiles.columns):
        ax.plot(voltage_profiles.index, voltage_profiles[col],
                linewidth=2.5, marker="o", markersize=4, label=col)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Mean Utilisation (%)")
    ax.set_title("Data Centre Utilisation by Voltage Level")
    hour_labels(ax)
    ax.set_ylim(0, 105)
    ax.legend()
    save_fig(fig, os.path.join(CHART_DIR, "02_dc_voltage_comparison.png"))

# ── 2d. Flexibility by hour ──────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(profile.index, profile["flexibility_pct"],
       color=COLORS["green"], alpha=0.85, edgecolor="white")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Headroom (% of max capacity)")
ax.set_title(f"Data Centre Flexibility by Hour — {data_label}")
hour_labels(ax)
save_fig(fig, os.path.join(CHART_DIR, "02_dc_flexibility_by_hour.png"))

# ── 2e. Month-hour heatmap ───────────────────────────────────────────────────

if monthly_hourly_pivot is not None:
    month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                   7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    fig, ax = plt.subplots(figsize=(12, 7))
    cols = [c for c in monthly_hourly_pivot.columns if c in month_names]
    plot_data = monthly_hourly_pivot[cols]
    im = ax.imshow(plot_data.values, aspect="auto", cmap="Blues", interpolation="nearest")
    ax.set_yticks(range(24))
    ax.set_yticklabels([f"{h:02d}:00" for h in range(24)])
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([month_names[c] for c in cols])
    ax.set_ylabel("Hour of Day")
    ax.set_title("DC Utilisation Heatmap — Month × Hour (%)")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Utilisation (%)")
    save_fig(fig, os.path.join(CHART_DIR, "02_dc_heatmap_month_hour.png"))

# ── 2f. Data coverage timeline ───────────────────────────────────────────────

if USE_REAL_DATA:
    fig, ax = plt.subplots(figsize=(12, 3))
    dc_dates = dc_raw.groupby(dc_raw["datetime"].dt.date).size()
    ax.fill_between(pd.to_datetime(dc_dates.index), 0, 1,
                    alpha=0.3, color=COLORS["blue"], label="UKPN DC data available")
    ax.axvspan(BOA_START, BOA_END, alpha=0.15,
               color=COLORS["red"], label="BOA curtailment period (Apr 2024 – Mar 2026)")
    ax.set_yticks([])
    ax.set_xlabel("Date")
    ax.set_title("Data Coverage: UKPN DC Profiles vs BOA Curtailment Period")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %Y"))
    save_fig(fig, os.path.join(CHART_DIR, "02_data_coverage.png"))

print(f"\n  ✓ All charts saved to {CHART_DIR}/02_*.png")
print("  Done.\n")
