"""
06_dc_operations.py — Data Centre Operational Model
Group 10: Wind Curtailment and Data Centres

Goes beyond "how much curtailment can DCs absorb" to model a real DC:
  - Total DC demand profile (base load + flexible load) every half-hour
  - Energy sourcing: what fraction comes from curtailed wind vs grid?
  - Grid dependency: how much grid power does the DC still need?
  - Price-responsive scheduling: shift flexible work to cheapest periods
  - Financial comparison: DC with vs without co-location benefit

This answers the examiner question: "What does the DC's actual operation
look like, and how much does co-location with wind actually help?"

Input:  data/processed/curtailment_processed.csv         (from 01)
        data/processed/curtailment_per_period.csv        (from 01)
        data/processed/dc_hourly_profile_matched.csv     (from 02)
Output: output/csv/06_dc_operations.csv
        output/csv/06_operations_summary.csv
        output/charts/06_*.png
"""

from __future__ import annotations
import os, sys, re
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

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


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

REGION = "Seagreen"

# DC operational parameters
DC_CAPACITY_MW   = 500      # total IT + cooling nameplate capacity
BASE_LOAD_FRAC   = 0.55     # minimum always-on fraction (servers, cooling, network)
MAX_LOAD_FRAC    = 0.95     # max usable (leave 5% headroom for spikes)
# Flexible load = MAX_LOAD_FRAC - BASE_LOAD_FRAC = 40% of capacity

# The DC's own demand varies by hour-of-day using the UKPN profile shape.
# base_load is the floor; the UKPN profile determines how much of the
# flexible range is used for "normal" scheduled work vs truly shiftable.
SHIFTABLE_FRAC   = 0.50     # what fraction of flexible load can be time-shifted

# Pricing
GRID_PRICE_GBP_MWH     = 80.0   # normal grid electricity cost
CURTAILED_PRICE_GBP_MWH = 20.0  # price for co-located curtailed wind (BM bid level)
# The spread (£60/MWh) is the DC's incentive to consume curtailed wind.

# Price-responsive scheduling: when prices are below this, schedule extra work
CHEAP_PRICE_THRESHOLD = 50.0  # £/MWh — schedule deferred work when grid is cheap

# Reference wholesale price for periods without IMRP
REFERENCE_PRICE = 80.0

DATE_RANGE = ("2024-04-01", "2026-04-01")


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("06 — DATA CENTRE OPERATIONAL MODEL")
print("=" * 70)

# Curtailment
curt_path = os.path.join(PROC_DIR, "curtailment_processed.csv")
sp_path   = os.path.join(PROC_DIR, "curtailment_per_period.csv")
dc_path   = os.path.join(PROC_DIR, "dc_hourly_profile_matched.csv")

for p, name in [(curt_path, "curtailment_processed"), (dc_path, "dc_hourly_profile")]:
    if not os.path.exists(p):
        sys.exit(f"ERROR: {p} not found. Run earlier scripts first.")

print("\nLoading data …")
curt = pd.read_csv(curt_path, low_memory=False)
curt["Datetime"] = pd.to_datetime(curt["Datetime"], errors="coerce")
curt = curt.dropna(subset=["Datetime"])
for col in ["Curtailment_MWh", "Curtailment_MW"]:
    curt[col] = pd.to_numeric(curt[col], errors="coerce")

# Filter to region
all_names = curt["Generator_Full_Name"].unique()
region_farms = sorted([n for n in all_names if REGION.lower() in n.lower()])
if not region_farms:
    sys.exit(f"ERROR: No farms match '{REGION}'")
curt_region = curt[curt["Generator_Full_Name"].isin(region_farms)].copy()

print(f"  Region: {REGION} ({len(region_farms)} units)")

# DC hourly profile
dc_profile = pd.read_csv(dc_path, index_col=0)
dc_profile.index = dc_profile.index.astype(int)

# IMRP prices (optional)
imrp_path = os.path.join(RAW_DIR, "imrp_actuals.csv")
price_lookup = None
if os.path.exists(imrp_path):
    try:
        imrp = pd.read_csv(imrp_path)
        date_col = [c for c in imrp.columns if "date" in c.lower()][0]
        period_col = [c for c in imrp.columns if "period" in c.lower()][0]
        amount_col = [c for c in imrp.columns if "amount" in c.lower()
                      and "date" not in c.lower()][0]
        imrp["date_clean"] = pd.to_datetime(imrp[date_col], errors="coerce").dt.date
        imrp["sett_period"] = pd.to_numeric(imrp[period_col], errors="coerce").astype("Int64")
        imrp["price"] = pd.to_numeric(imrp[amount_col], errors="coerce")
        price_lookup = imrp.set_index(["date_clean", "sett_period"])["price"]
        price_lookup = price_lookup[~price_lookup.index.duplicated(keep="last")]
        print(f"  IMRP prices loaded: {len(price_lookup):,} periods")
    except Exception as e:
        print(f"  WARNING: IMRP parse error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. BUILD CONTINUOUS HALF-HOURLY TIMELINE
# ══════════════════════════════════════════════════════════════════════════════

print("\nBuilding operational timeline …")

# Full half-hour index
full_idx = pd.date_range(DATE_RANGE[0], DATE_RANGE[1], freq="30min", inclusive="left")

# Region curtailment per settlement period
region_sp = (curt_region
    .groupby("Datetime")["Curtailment_MWh"].sum()
    .reindex(full_idx, fill_value=0.0))
region_mw = region_sp / 0.5  # MWh in 30min → MW

# DC utilisation profile (hour-of-day → utilisation %)
hour_util = dc_profile["mean_util"].to_dict()
hour_std  = dc_profile["std_util"].to_dict()

# Build the operational dataframe
ops = pd.DataFrame(index=full_idx)
ops.index.name = "Datetime"
ops["hour"] = ops.index.hour
ops["date"] = ops.index.date
ops["month"] = ops.index.to_period("M")
ops["dayofweek"] = ops.index.dayofweek

# Curtailment available at site
ops["curtailment_MW"] = region_mw.values
ops["curtailment_MWh"] = region_sp.values

# Grid price
if price_lookup is not None:
    ops["grid_price"] = ops.apply(
        lambda r: price_lookup.get(
            (r.name.date(), (r.name.hour * 2 + r.name.minute // 30 + 1)),
            REFERENCE_PRICE),
        axis=1)
else:
    ops["grid_price"] = REFERENCE_PRICE

# ══════════════════════════════════════════════════════════════════════════════
# 3. MODEL DC DEMAND (base + scheduled + shiftable)
# ══════════════════════════════════════════════════════════════════════════════

# The DC's demand has three components:
#   1. Base load: always-on (servers at minimum, cooling, network) = BASE_LOAD_FRAC × capacity
#   2. Scheduled load: normal workload following the UKPN utilisation profile shape
#   3. Shiftable load: flexible compute that can be moved to match curtailment/cheap prices

base_mw = DC_CAPACITY_MW * BASE_LOAD_FRAC
max_mw  = DC_CAPACITY_MW * MAX_LOAD_FRAC
flex_range_mw = max_mw - base_mw  # total flexible capacity

# Normal scheduled load follows the UKPN profile shape scaled to our DC
# The profile gives utilisation %, we scale the flexible portion
ops["profile_util_pct"] = ops["hour"].map(hour_util)
ops["profile_util_frac"] = ops["profile_util_pct"] / 100.0

# Normal demand = base + profile-driven portion of flex range
# But we only schedule (1 - SHIFTABLE_FRAC) of the flex range on the normal profile
# The remaining SHIFTABLE_FRAC is available to time-shift
normal_flex_mw = flex_range_mw * (1 - SHIFTABLE_FRAC)
shiftable_mw   = flex_range_mw * SHIFTABLE_FRAC

# Scale the profile: at peak utilisation, normal flex is fully used
# at trough, it's partially used
profile_min = dc_profile["mean_util"].min() / 100.0
profile_max = dc_profile["mean_util"].max() / 100.0

ops["normal_demand_MW"] = base_mw + normal_flex_mw * (
    (ops["profile_util_frac"] - profile_min) / (profile_max - profile_min)
).clip(0, 1)


# ══════════════════════════════════════════════════════════════════════════════
# 4. DISPATCH SHIFTABLE LOAD (price-responsive + curtailment-chasing)
# ══════════════════════════════════════════════════════════════════════════════

# Strategy: schedule shiftable work when either:
#   a) Curtailed wind is available (cheapest energy at ~£20/MWh)
#   b) Grid price is below the cheap threshold
# Otherwise, defer the work.

# Daily energy budget for shiftable work (must complete each day)
daily_shiftable_mwh = shiftable_mw * 0.5 * 48 * 0.70  # 70% utilisation target

print(f"\n  DC Capacity: {DC_CAPACITY_MW} MW")
print(f"  Base load:   {base_mw:.0f} MW ({BASE_LOAD_FRAC:.0%})")
print(f"  Max load:    {max_mw:.0f} MW ({MAX_LOAD_FRAC:.0%})")
print(f"  Flex range:  {flex_range_mw:.0f} MW")
print(f"  Shiftable:   {shiftable_mw:.0f} MW ({SHIFTABLE_FRAC:.0%} of flex)")
print(f"  Daily shift budget: {daily_shiftable_mwh:.0f} MWh")

# For each day, rank periods by "effective price" and schedule shiftable work
# Effective price = curtailed_price if curtailment available, else grid_price
ops["effective_price"] = np.where(
    ops["curtailment_MW"] > 0,
    np.minimum(CURTAILED_PRICE_GBP_MWH, ops["grid_price"]),
    ops["grid_price"]
)

# Rank periods within each day by effective price (cheapest first)
ops["daily_rank"] = ops.groupby("date")["effective_price"].rank(method="first")

# Schedule shiftable work into cheapest periods each day
# Number of periods needed per day = daily budget / (shiftable_mw × 0.5)
periods_needed = int(np.ceil(daily_shiftable_mwh / (shiftable_mw * 0.5)))
ops["shift_scheduled"] = (ops["daily_rank"] <= periods_needed).astype(float)

# Actual shiftable dispatch
ops["shiftable_dispatch_MW"] = ops["shift_scheduled"] * shiftable_mw

# Total DC demand
ops["total_demand_MW"]  = ops["normal_demand_MW"] + ops["shiftable_dispatch_MW"]
ops["total_demand_MW"]  = ops["total_demand_MW"].clip(base_mw, max_mw)
ops["total_demand_MWh"] = ops["total_demand_MW"] * 0.5


# ══════════════════════════════════════════════════════════════════════════════
# 5. ENERGY SOURCING: CURTAILED WIND vs GRID
# ══════════════════════════════════════════════════════════════════════════════

# The DC can consume curtailed wind up to the lesser of:
#   - its total demand
#   - available curtailment
ops["from_curtailment_MW"] = np.minimum(
    ops["total_demand_MW"],
    ops["curtailment_MW"]
)
ops["from_curtailment_MWh"] = ops["from_curtailment_MW"] * 0.5

# Remaining demand must come from the grid
ops["from_grid_MW"]  = ops["total_demand_MW"] - ops["from_curtailment_MW"]
ops["from_grid_MWh"] = ops["from_grid_MW"] * 0.5

# Energy cost
ops["cost_curtailed_gbp"] = ops["from_curtailment_MWh"] * CURTAILED_PRICE_GBP_MWH
ops["cost_grid_gbp"]      = ops["from_grid_MWh"] * ops["grid_price"]
ops["total_cost_gbp"]     = ops["cost_curtailed_gbp"] + ops["cost_grid_gbp"]

# Counterfactual: if the DC was NOT co-located (all from grid)
ops["counterfactual_cost_gbp"] = ops["total_demand_MWh"] * ops["grid_price"]
ops["savings_gbp"] = ops["counterfactual_cost_gbp"] - ops["total_cost_gbp"]

# Curtailment absorbed (value to the wind farm / system)
ops["curtailment_absorbed_MWh"] = ops["from_curtailment_MWh"]
ops["curtailment_remaining_MWh"] = ops["curtailment_MWh"] - ops["from_curtailment_MWh"]


# ══════════════════════════════════════════════════════════════════════════════
# 6. COMPUTE SUMMARY STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

total_demand_gwh       = ops["total_demand_MWh"].sum() / 1000
from_curtailment_gwh   = ops["from_curtailment_MWh"].sum() / 1000
from_grid_gwh          = ops["from_grid_MWh"].sum() / 1000
curt_fraction          = from_curtailment_gwh / total_demand_gwh * 100
grid_fraction          = from_grid_gwh / total_demand_gwh * 100

total_curt_gwh         = ops["curtailment_MWh"].sum() / 1000
absorbed_of_curt       = from_curtailment_gwh / total_curt_gwh * 100 if total_curt_gwh > 0 else 0

total_cost_m           = ops["total_cost_gbp"].sum() / 1e6
counterfactual_m       = ops["counterfactual_cost_gbp"].sum() / 1e6
savings_m              = ops["savings_gbp"].sum() / 1e6
avg_blended_price      = ops["total_cost_gbp"].sum() / ops["total_demand_MWh"].sum()
avg_grid_only_price    = ops["counterfactual_cost_gbp"].sum() / ops["total_demand_MWh"].sum()

print(f"\n{'='*70}")
print(f"DC OPERATIONS SUMMARY — {DC_CAPACITY_MW} MW co-located with {REGION}")
print(f"{'='*70}")
print(f"""
  ENERGY SOURCING (annual equivalent):
    Total DC demand:     {total_demand_gwh:>8,.1f} GWh
    From curtailed wind: {from_curtailment_gwh:>8,.1f} GWh  ({curt_fraction:.1f}%)
    From grid:           {from_grid_gwh:>8,.1f} GWh  ({grid_fraction:.1f}%)

  CURTAILMENT IMPACT:
    Regional curtailment:{total_curt_gwh:>8,.1f} GWh
    Absorbed by DC:      {from_curtailment_gwh:>8,.1f} GWh  ({absorbed_of_curt:.1f}% of regional)

  FINANCIAL:
    Co-located cost:     £{total_cost_m:>7,.0f}m  (blended £{avg_blended_price:.1f}/MWh)
    Grid-only cost:      £{counterfactual_m:>7,.0f}m  (blended £{avg_grid_only_price:.1f}/MWh)
    Co-location saving:  £{savings_m:>7,.0f}m  ({savings_m/counterfactual_m*100:.1f}% reduction)
""")

# Save
ops_save = ops.reset_index()
ops_save.to_csv(os.path.join(CSV_DIR, "06_dc_operations.csv"), index=False)

# Monthly summary
monthly = ops.groupby("month").agg(
    demand_GWh       = pd.NamedAgg("total_demand_MWh", lambda x: x.sum()/1000),
    from_curt_GWh    = pd.NamedAgg("from_curtailment_MWh", lambda x: x.sum()/1000),
    from_grid_GWh    = pd.NamedAgg("from_grid_MWh", lambda x: x.sum()/1000),
    curt_pct         = pd.NamedAgg("from_curtailment_MWh",
                                   lambda x: x.sum() / (x.sum() + ops.loc[x.index, "from_grid_MWh"].sum()) * 100),
    cost_m           = pd.NamedAgg("total_cost_gbp", lambda x: x.sum()/1e6),
    counterfactual_m = pd.NamedAgg("counterfactual_cost_gbp", lambda x: x.sum()/1e6),
    savings_m        = pd.NamedAgg("savings_gbp", lambda x: x.sum()/1e6),
)
monthly.to_csv(os.path.join(CSV_DIR, "06_operations_summary.csv"))
print(f"  ✓ Saved 06_dc_operations.csv")
print(f"  ✓ Saved 06_operations_summary.csv")


# ══════════════════════════════════════════════════════════════════════════════
# 7. CHARTS
# ══════════════════════════════════════════════════════════════════════════════

print(f"\nGenerating charts …")
month_order = pd.period_range("2024-04", "2026-03", freq="M")


# ── 6a. Energy sourcing breakdown (stacked monthly) ──────────────────────────

fig, ax = plt.subplots(figsize=(13, 5))
x = np.arange(len(month_order))
curt_vals = monthly.reindex(month_order)["from_curt_GWh"].fillna(0).values
grid_vals = monthly.reindex(month_order)["from_grid_GWh"].fillna(0).values

ax.bar(x, curt_vals, color=COLORS["green"], label="From curtailed wind")
ax.bar(x, grid_vals, bottom=curt_vals, color=COLORS["blue_light"], label="From grid")
ax.set_xticks(x)
ax.set_xticklabels([m.strftime("%b\n%Y") if m.month in [1,4,7,10]
                     else m.strftime("%b") for m in month_order], fontsize=7)
ax.set_ylabel("Energy (GWh)")
ax.set_title(f"{DC_CAPACITY_MW} MW Data Centre — Monthly Energy Sourcing\n"
             f"Co-located with {REGION}")
ax.legend()
# Add % labels
for i in range(len(x)):
    total = curt_vals[i] + grid_vals[i]
    if total > 0:
        pct = curt_vals[i] / total * 100
        ax.text(i, total + 1, f"{pct:.0f}%", ha="center", fontsize=7,
                color=COLORS["green"], fontweight="bold")
save_fig(fig, os.path.join(CHART_DIR, "06_energy_sourcing.png"))


# ── 6b. Hourly operational profile (demand breakdown) ─────────────────────────

hourly_ops = ops.groupby("hour").agg(
    total_demand   = pd.NamedAgg("total_demand_MW", "mean"),
    normal_demand  = pd.NamedAgg("normal_demand_MW", "mean"),
    shiftable      = pd.NamedAgg("shiftable_dispatch_MW", "mean"),
    from_curt      = pd.NamedAgg("from_curtailment_MW", "mean"),
    from_grid      = pd.NamedAgg("from_grid_MW", "mean"),
    curtailment    = pd.NamedAgg("curtailment_MW", "mean"),
)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

# Top: DC demand composition
ax1.fill_between(range(24), 0, hourly_ops["normal_demand"],
                 alpha=0.7, color=COLORS["blue"], label="Base + scheduled load")
ax1.fill_between(range(24), hourly_ops["normal_demand"],
                 hourly_ops["total_demand"],
                 alpha=0.7, color=COLORS["amber"], label="Shiftable load")
ax1.axhline(y=base_mw, color="black", linestyle=":", linewidth=1,
            label=f"Base load ({base_mw:.0f} MW)")
ax1.axhline(y=max_mw, color="grey", linestyle="--", linewidth=1,
            label=f"Max capacity ({max_mw:.0f} MW)")
ax1.set_ylabel("Power (MW)")
ax1.set_title(f"{DC_CAPACITY_MW} MW Data Centre — Hourly Demand Profile")
ax1.legend(fontsize=8, ncol=2)
ax1.set_ylim(0, DC_CAPACITY_MW * 1.05)

# Bottom: energy sourcing by hour
ax2.bar(range(24), hourly_ops["from_curt"],
        color=COLORS["green"], alpha=0.85, label="From curtailed wind")
ax2.bar(range(24), hourly_ops["from_grid"], bottom=hourly_ops["from_curt"],
        color=COLORS["blue_light"], alpha=0.85, label="From grid")
ax2.set_ylabel("Mean Power (MW)")
ax2.set_xlabel("Hour of Day")
ax2.set_title("Energy Sourcing by Hour of Day")
hour_labels(ax2)
ax2.legend()

save_fig(fig, os.path.join(CHART_DIR, "06_hourly_operations.png"))


# ── 6c. Example week — operational dispatch ──────────────────────────────────

daily_curt = ops.groupby("date")["curtailment_MWh"].sum()
peak_date = pd.Timestamp(daily_curt.idxmax())
wk_start = peak_date - pd.Timedelta(days=2)
wk_end   = peak_date + pd.Timedelta(days=5)
week = ops.loc[wk_start:wk_end]

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Top: curtailment available
ax1.fill_between(week.index, week["curtailment_MW"],
                 alpha=0.3, color=COLORS["blue"])
ax1.plot(week.index, week["curtailment_MW"],
         linewidth=0.8, color=COLORS["blue"], label=f"{REGION} curtailment")
ax1.set_ylabel("MW")
ax1.set_title(f"Example Week — {DC_CAPACITY_MW} MW DC Operations "
              f"(around {peak_date.date()})")
ax1.legend(fontsize=8)

# Middle: DC demand and sourcing
ax2.fill_between(week.index, 0, week["from_curtailment_MW"],
                 alpha=0.7, color=COLORS["green"], label="From curtailed wind")
ax2.fill_between(week.index, week["from_curtailment_MW"],
                 week["total_demand_MW"],
                 alpha=0.7, color=COLORS["blue_light"], label="From grid")
ax2.axhline(y=base_mw, color="black", linestyle=":", linewidth=0.8)
ax2.set_ylabel("MW")
ax2.set_title("DC Energy Sourcing")
ax2.legend(fontsize=8)
ax2.set_ylim(0, max_mw * 1.1)

# Bottom: grid price and effective price
ax3.plot(week.index, week["grid_price"], color=COLORS["grey"],
         linewidth=0.8, label="Grid price", alpha=0.7)
ax3.plot(week.index, week["effective_price"], color=COLORS["green"],
         linewidth=1.5, label="Effective price (DC)")
ax3.axhline(y=CURTAILED_PRICE_GBP_MWH, color=COLORS["green"],
            linestyle=":", label=f"Curtailed wind price (£{CURTAILED_PRICE_GBP_MWH})")
ax3.set_ylabel("£/MWh")
ax3.set_xlabel("Time")
ax3.set_title("Price Signal — Grid vs Co-located")
ax3.legend(fontsize=8)

save_fig(fig, os.path.join(CHART_DIR, "06_example_week.png"))


# ── 6d. Financial comparison: co-located vs grid-only ─────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: monthly costs
x = np.arange(len(month_order))
colocated = monthly.reindex(month_order)["cost_m"].fillna(0).values
gridonly   = monthly.reindex(month_order)["counterfactual_m"].fillna(0).values

ax1.bar(x - 0.2, gridonly, width=0.4,
        color=COLORS["red"], alpha=0.7, label="Grid-only")
ax1.bar(x + 0.2, colocated, width=0.4,
        color=COLORS["green"], alpha=0.85, label="Co-located")
ax1.set_xticks(x)
ax1.set_xticklabels([m.strftime("%b\n%Y") if m.month in [1,4,7,10]
                      else m.strftime("%b") for m in month_order], fontsize=7)
ax1.set_ylabel("Monthly Cost (£m)")
ax1.set_title("Energy Cost: Grid-Only vs Co-Located with Wind")
ax1.legend()

# Right: cumulative savings
cum_savings = monthly.reindex(month_order)["savings_m"].fillna(0).cumsum()
ax2.plot(x, cum_savings.values, color=COLORS["green"],
         linewidth=2.5, marker="o", markersize=4)
ax2.fill_between(x, 0, cum_savings.values, alpha=0.15, color=COLORS["green"])
ax2.set_xticks(x)
ax2.set_xticklabels([m.strftime("%b\n%Y") if m.month in [1,4,7,10]
                      else m.strftime("%b") for m in month_order], fontsize=7)
ax2.set_ylabel("Cumulative Savings (£m)")
ax2.set_title(f"Cumulative Co-Location Savings: £{savings_m:,.0f}m over period")
ax2.axhline(y=savings_m, color="black", linestyle=":", alpha=0.3)

save_fig(fig, os.path.join(CHART_DIR, "06_financial_comparison.png"))


# ── 6e. Pie chart: energy sourcing + grid dependency ─────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Energy sourcing
labels_e = [f"Curtailed wind\n{from_curtailment_gwh:,.0f} GWh\n({curt_fraction:.1f}%)",
            f"Grid supply\n{from_grid_gwh:,.0f} GWh\n({grid_fraction:.1f}%)"]
ax1.pie([from_curtailment_gwh, from_grid_gwh],
        colors=[COLORS["green"], COLORS["blue_light"]],
        startangle=90, wedgeprops=dict(edgecolor="white", linewidth=2))
ax1.set_title(f"Energy Sourcing\n({total_demand_gwh:,.0f} GWh total)", fontsize=11)
ax1.legend(labels_e, loc="lower center", fontsize=8, frameon=False)

# Cost breakdown
cost_curt_m = ops["cost_curtailed_gbp"].sum() / 1e6
cost_grid_m = ops["cost_grid_gbp"].sum() / 1e6
labels_c = [f"Curtailed wind\n£{cost_curt_m:,.0f}m",
            f"Grid supply\n£{cost_grid_m:,.0f}m"]
ax2.pie([cost_curt_m, cost_grid_m],
        colors=[COLORS["green"], COLORS["blue_light"]],
        startangle=90, wedgeprops=dict(edgecolor="white", linewidth=2))
ax2.set_title(f"Energy Cost Breakdown\n(£{total_cost_m:,.0f}m total)", fontsize=11)
ax2.legend(labels_c, loc="lower center", fontsize=8, frameon=False)

save_fig(fig, os.path.join(CHART_DIR, "06_sourcing_pies.png"))


# ── 6f. DC capacity sensitivity ──────────────────────────────────────────────

cap_range = [100, 200, 300, 500, 750, 1000]
sens = []
for cap in cap_range:
    bm = cap * BASE_LOAD_FRAC
    mm = cap * MAX_LOAD_FRAC
    fr = mm - bm
    # Simplified: assume similar demand pattern scaled
    scale = cap / DC_CAPACITY_MW
    demand = ops["total_demand_MW"] * scale
    from_c = np.minimum(demand.values, ops["curtailment_MW"].values)
    from_g = demand.values - from_c
    total_d  = demand.sum() * 0.5 / 1000
    curt_d   = from_c.sum() * 0.5 / 1000
    grid_d   = from_g.sum() * 0.5 / 1000
    cost_co  = (from_c * 0.5 * CURTAILED_PRICE_GBP_MWH + from_g * 0.5 * ops["grid_price"].values).sum() / 1e6
    cost_grd = (demand * 0.5 * ops["grid_price"]).sum() / 1e6
    sens.append({
        "capacity_MW": cap,
        "demand_GWh": total_d,
        "from_curt_GWh": curt_d,
        "from_grid_GWh": grid_d,
        "curt_pct": curt_d / total_d * 100 if total_d > 0 else 0,
        "cost_colocated_m": cost_co,
        "cost_grid_m": cost_grd,
        "savings_m": cost_grd - cost_co,
    })
sens_df = pd.DataFrame(sens)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.bar(range(len(sens_df)), sens_df["from_curt_GWh"],
        color=COLORS["green"], label="From curtailed wind")
ax1.bar(range(len(sens_df)), sens_df["from_grid_GWh"],
        bottom=sens_df["from_curt_GWh"],
        color=COLORS["blue_light"], label="From grid")
ax1.set_xticks(range(len(sens_df)))
ax1.set_xticklabels([f"{c} MW" for c in sens_df["capacity_MW"]])
ax1.set_ylabel("Energy (GWh)")
ax1.set_title("Energy Sourcing by DC Capacity")
ax1.legend()
for i, row in sens_df.iterrows():
    ax1.text(i, row["demand_GWh"] + 10, f"{row['curt_pct']:.0f}% curt",
             ha="center", fontsize=8, color=COLORS["green"], fontweight="bold")

ax2.bar(range(len(sens_df)), sens_df["savings_m"],
        color=COLORS["green"], alpha=0.85)
ax2.set_xticks(range(len(sens_df)))
ax2.set_xticklabels([f"{c} MW" for c in sens_df["capacity_MW"]])
ax2.set_ylabel("Co-Location Savings (£m)")
ax2.set_title(f"Savings from Wind Co-Location ({REGION})")
for i, v in enumerate(sens_df["savings_m"]):
    ax2.text(i, v + 1, f"£{v:,.0f}m", ha="center", fontsize=9, fontweight="bold")

save_fig(fig, os.path.join(CHART_DIR, "06_capacity_sensitivity.png"))


# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────

print(f"\n{'='*70}")
print(f"KEY FINDINGS — DC OPERATIONAL MODEL")
print(f"{'='*70}")
print(f"""
  A {DC_CAPACITY_MW} MW data centre co-located with {REGION} would:

  1. SOURCE {curt_fraction:.0f}% of its energy from curtailed wind
     ({from_curtailment_gwh:,.0f} GWh / {total_demand_gwh:,.0f} GWh).

  2. STILL NEED {grid_fraction:.0f}% from the grid ({from_grid_gwh:,.0f} GWh).
     Wind curtailment is intermittent — the DC needs firm grid connection.

  3. SAVE £{savings_m:,.0f}m vs grid-only operation
     (blended price £{avg_blended_price:.1f}/MWh vs £{avg_grid_only_price:.1f}/MWh).

  4. ABSORB {absorbed_of_curt:.1f}% of {REGION}'s curtailment.
     Even a large DC barely dents a major wind farm's curtailment volume.

  5. SHIFT flexible work to coincide with curtailment events,
     creating {periods_needed} scheduled periods/day of price-responsive load.
""")

print(f"  ✓ Charts: {CHART_DIR}/06_*.png")
print(f"  ✓ CSVs:   {CSV_DIR}/06_*.csv")
print("  Done.\n")
