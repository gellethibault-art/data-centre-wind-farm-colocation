"""
05_financial_analysis.py — Financial Impact of Wind Curtailment
Group 10: Wind Curtailment and Data Centres

Quantifies the financial cost of curtailment and the potential savings
from DC-based absorption, filtered by wind farm "region" (e.g. Seagreen).

Region filtering uses partial name matching on Generator_Full_Name, so
REGION = "Seagreen" will capture Seagreen 1, Seagreen 2, … Seagreen 6.

Supports multiple pricing datasets (loaded if present, skipped if not):
  - IMRP half-hourly prices  (best: actual settlement period prices)
  - Balancing volumes/costs  (NESO MBSS quarterly files)
  - BS_NETBSD files          (Net BSAD, messy headers — auto-cleaned)
  - CfD generation/payments  (LCCC daily data for project-level context)

If no pricing data is available, uses an assumed reference wholesale price
as a proxy (configurable below). This gives a lower bound on value.

Input:  data/processed/curtailment_processed.csv       (from 01)
        data/processed/curtailment_per_period.csv      (from 01)
        data/raw/imrp_actuals.csv                      (optional)
        data/raw/q1_vol_fy*.csv                        (optional)
        data/raw/q1_cost_fy*.csv                       (optional)
        data/raw/BS_NETBSD_*.csv                       (optional)
        data/raw/actual_cfd_generation_*.csv            (optional)
Output: output/csv/05_region_curtailment.csv
        output/csv/05_financial_summary.csv
        output/csv/05_merged_halfhour.csv               (if pricing data)
        output/charts/05_*.png
"""

from __future__ import annotations
import os, sys, glob, re
from typing import Optional, List, Dict

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
# CONFIG — edit these
# ══════════════════════════════════════════════════════════════════════════════

# Region to analyse (partial match on Generator_Full_Name, case-insensitive)
# Set to None to analyse ALL wind farms
REGION = "Seagreen"

# Fallback wholesale reference price (£/MWh) when IMRP data unavailable.
# GB day-ahead baseload averaged ~£70–90/MWh in 2024-25.
REFERENCE_PRICE_GBP_MWH = 80.0

# DC fleet assumptions (for "absorption value" calculation)
DC_FLEET_MW   = 1000
DC_FLEX_FRAC  = 0.50
DC_HEADROOM   = 0.30   # approximate average headroom fraction

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD BOA CURTAILMENT DATA
# ══════════════════════════════════════════════════════════════════════════════

curt_path = os.path.join(PROC_DIR, "curtailment_processed.csv")
sp_path   = os.path.join(PROC_DIR, "curtailment_per_period.csv")

if not os.path.exists(curt_path):
    sys.exit("ERROR: Run 01_curtailment_analysis.py first to generate curtailment_processed.csv")

print("=" * 70)
print("05 — FINANCIAL IMPACT ANALYSIS")
print("=" * 70)

print("\nLoading curtailment data …")
curt = pd.read_csv(curt_path, low_memory=False)
curt["Date"] = pd.to_datetime(curt["Date"], errors="coerce")
curt["Datetime"] = pd.to_datetime(curt["Datetime"], errors="coerce")
curt = curt.dropna(subset=["Datetime"])

# Ensure numeric
for col in ["Curtailment_MWh", "Curtailment_MW", "BOA_Volume"]:
    if col in curt.columns:
        curt[col] = pd.to_numeric(curt[col], errors="coerce")

print(f"  Total events: {len(curt):,}")
print(f"  Unique farms: {curt['Generator_Full_Name'].nunique()}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. REGION FILTERING
# ══════════════════════════════════════════════════════════════════════════════

def find_region_farms(df: pd.DataFrame, region: str) -> List[str]:
    """Find all Generator_Full_Name values matching a region prefix."""
    all_names = df["Generator_Full_Name"].unique()
    matches = sorted([n for n in all_names
                      if region.lower() in n.lower()])
    return matches


def find_region_bmu_ids(df: pd.DataFrame, region: str) -> List[str]:
    """Find all Generator_Name (BMU IDs) matching a region."""
    farms = find_region_farms(df, region)
    mask = df["Generator_Full_Name"].isin(farms)
    return sorted(df.loc[mask, "Generator_Name"].unique())


if REGION:
    region_farms = find_region_farms(curt, REGION)
    region_bmus  = find_region_bmu_ids(curt, REGION)
    if not region_farms:
        print(f"\n  WARNING: No farms match region '{REGION}'.")
        print(f"  Available regions (multi-unit):")
        names = curt["Generator_Full_Name"].unique()
        bases = sorted(set(re.sub(r"\s*\d+$", "", n).strip() for n in names))
        for b in bases:
            cnt = sum(1 for n in names if n.startswith(b))
            if cnt > 1:
                print(f"    {b} ({cnt} units)")
        sys.exit(1)

    region_mask = curt["Generator_Full_Name"].isin(region_farms)
    curt_region = curt[region_mask].copy()
    curt_system = curt.copy()

    print(f"\n  Region: {REGION}")
    print(f"  Matched farms ({len(region_farms)}):")
    for f in region_farms:
        bmu = curt.loc[curt["Generator_Full_Name"] == f, "Generator_Name"].iloc[0]
        n_events = region_mask.sum()
        total = curt.loc[curt["Generator_Full_Name"] == f, "Curtailment_MWh"].sum()
        print(f"    {bmu:12s}  {f:25s}  {total/1000:8,.1f} GWh")
    print(f"  BMU IDs: {region_bmus}")

    region_total_gwh = curt_region["Curtailment_MWh"].sum() / 1000
    system_total_gwh = curt_system["Curtailment_MWh"].sum() / 1000
    print(f"\n  Region curtailment: {region_total_gwh:,.1f} GWh "
          f"({region_total_gwh/system_total_gwh*100:.1f}% of system total)")
else:
    curt_region = curt.copy()
    curt_system = curt.copy()
    region_farms = list(curt["Generator_Full_Name"].unique())
    region_bmus  = list(curt["Generator_Name"].unique())
    region_total_gwh = curt["Curtailment_MWh"].sum() / 1000
    system_total_gwh = region_total_gwh
    print("  Analysing ALL wind farms (no region filter)")

REGION_LABEL = REGION or "All Farms"


# ══════════════════════════════════════════════════════════════════════════════
# 3. BUILD REGION HALF-HOURLY TIME SERIES
# ══════════════════════════════════════════════════════════════════════════════

# Aggregate per settlement period for the region
region_sp = (curt_region
    .groupby(["Date", "Settlement_Period", "Datetime"])
    .agg(
        curtailment_MWh=("Curtailment_MWh", "sum"),
        curtailment_MW=("Curtailment_MW", "sum"),
        n_farms=("Generator_Name", "nunique"),
        boa_volume=("BOA_Volume", "sum"),
    )
    .reset_index()
)
region_sp["date_only"] = region_sp["Date"].dt.date
region_sp["HourInt"]   = region_sp["Datetime"].dt.hour
region_sp["Month"]     = region_sp["Date"].dt.to_period("M")

# Same for system-wide (for comparison)
system_sp = (curt_system
    .groupby(["Date", "Settlement_Period", "Datetime"])
    .agg(curtailment_MWh=("Curtailment_MWh", "sum"))
    .reset_index()
)
system_sp["Month"] = system_sp["Date"].dt.to_period("M")

print(f"\n  Region settlement periods with curtailment: {len(region_sp):,}")
print(f"  Mean curtailment per period: {region_sp['curtailment_MW'].mean():,.0f} MW")
print(f"  Peak curtailment period:     {region_sp['curtailment_MW'].max():,.0f} MW")


# ══════════════════════════════════════════════════════════════════════════════
# 4. LOAD PRICING DATASETS (optional — graceful fallback)
# ══════════════════════════════════════════════════════════════════════════════

pricing_source = "reference"  # will be upgraded if real data found
price_series = None           # will be a Series indexed by (date, sett_period)


# ── 4a. IMRP actuals (best half-hourly price) ────────────────────────────────

imrp_path = os.path.join(RAW_DIR, "imrp_actuals.csv")
if os.path.exists(imrp_path):
    print("\n  Loading IMRP price data …")
    try:
        imrp = pd.read_csv(imrp_path)
        # Clean date column (may contain timestamp suffix)
        date_col = [c for c in imrp.columns if "date" in c.lower()][0]
        period_col = [c for c in imrp.columns if "period" in c.lower()][0]
        amount_col = [c for c in imrp.columns if "amount" in c.lower() or "imrp" in c.lower()
                      and "date" not in c.lower()][0]

        imrp["date_clean"] = pd.to_datetime(imrp[date_col], errors="coerce").dt.date
        imrp["sett_period"] = pd.to_numeric(imrp[period_col], errors="coerce").astype("Int64")
        imrp["price_gbp_mwh"] = pd.to_numeric(imrp[amount_col], errors="coerce")
        imrp = imrp.dropna(subset=["date_clean", "sett_period", "price_gbp_mwh"])

        price_series = imrp.set_index(["date_clean", "sett_period"])["price_gbp_mwh"]
        price_series = price_series[~price_series.index.duplicated(keep="last")]
        pricing_source = "IMRP"
        print(f"    ✓ {len(imrp):,} half-hour prices loaded")
        print(f"    Date range: {imrp['date_clean'].min()} → {imrp['date_clean'].max()}")
        print(f"    Mean price: £{imrp['price_gbp_mwh'].mean():.2f}/MWh")
    except Exception as e:
        print(f"    WARNING: Could not parse IMRP data: {e}")
else:
    print("\n  ℹ  imrp_actuals.csv not found — using reference price")


# ── 4b. BS_NETBSD files (Net BSAD — messy headers) ───────────────────────────

bsad_files = sorted(glob.glob(os.path.join(RAW_DIR, "BS_NETBSD_*.csv")))
bsad_df = None

if bsad_files:
    print(f"\n  Loading BSAD data ({len(bsad_files)} files) …")
    frames = []
    for fpath in bsad_files:
        try:
            # Read raw to find the real header row
            raw = pd.read_csv(fpath, header=None, nrows=30, dtype=str)
            header_row = None
            for idx, row in raw.iterrows():
                row_str = row.astype(str).str.lower().str.strip()
                if row_str.str.contains("sett_date|settlement").any():
                    header_row = idx
                    break

            if header_row is None:
                print(f"    WARNING: Could not find header in {os.path.basename(fpath)}")
                continue

            df_b = pd.read_csv(fpath, skiprows=header_row, low_memory=False)
            # Clean column names
            df_b.columns = [c.strip().upper().replace(" ", "_") for c in df_b.columns]
            # Try to identify date and period columns
            date_col = next((c for c in df_b.columns if "DATE" in c), None)
            period_col = next((c for c in df_b.columns if "PERIOD" in c), None)
            if date_col and period_col:
                df_b["date_clean"] = pd.to_datetime(df_b[date_col], errors="coerce").dt.date
                df_b["sett_period"] = pd.to_numeric(df_b[period_col], errors="coerce").astype("Int64")
                df_b = df_b.dropna(subset=["date_clean", "sett_period"])
                frames.append(df_b)
                print(f"    ✓ {os.path.basename(fpath)}: {len(df_b):,} rows "
                      f"(header at row {header_row})")
        except Exception as e:
            print(f"    WARNING: {os.path.basename(fpath)}: {e}")

    if frames:
        bsad_df = pd.concat(frames, ignore_index=True)
        print(f"    Total BSAD rows: {len(bsad_df):,}")
else:
    print("\n  ℹ  No BS_NETBSD files found")


# ── 4c. Balancing volumes (q1_vol_*.csv) ─────────────────────────────────────

vol_files = sorted(glob.glob(os.path.join(RAW_DIR, "q*_vol_*.csv")))
vol_df = None

if vol_files:
    print(f"\n  Loading balancing volumes ({len(vol_files)} files) …")
    frames = []
    for fpath in vol_files:
        try:
            df_v = pd.read_csv(fpath)
            df_v.columns = [c.strip() for c in df_v.columns]
            frames.append(df_v)
            print(f"    ✓ {os.path.basename(fpath)}: {len(df_v):,} rows")
        except Exception as e:
            print(f"    WARNING: {os.path.basename(fpath)}: {e}")
    if frames:
        vol_df = pd.concat(frames, ignore_index=True)
        # Standardise columns
        col_map = {}
        for c in vol_df.columns:
            cl = c.lower()
            if "sett_date" in cl or "date" in cl:
                col_map[c] = "date_clean"
            elif "sett_period" in cl or "period" in cl:
                col_map[c] = "sett_period"
            elif "constraint" in cl and "offer" in cl:
                col_map[c] = "constraint_offers_mwh"
            elif "constraint" in cl and "bid" in cl:
                col_map[c] = "constraint_bids_mwh"
        vol_df.rename(columns=col_map, inplace=True)
        if "date_clean" in vol_df.columns:
            vol_df["date_clean"] = pd.to_datetime(vol_df["date_clean"], errors="coerce").dt.date
            vol_df["sett_period"] = pd.to_numeric(vol_df["sett_period"], errors="coerce").astype("Int64")
        print(f"    Total volume rows: {len(vol_df):,}")
else:
    print("\n  ℹ  No balancing volume files found")


# ── 4d. Balancing costs (q1_cost_*.csv) ──────────────────────────────────────

cost_files = sorted(glob.glob(os.path.join(RAW_DIR, "q*_cost_*.csv")))
cost_df = None

if cost_files:
    print(f"\n  Loading balancing costs ({len(cost_files)} files) …")
    frames = []
    for fpath in cost_files:
        try:
            df_c = pd.read_csv(fpath)
            df_c.columns = [c.strip() for c in df_c.columns]
            frames.append(df_c)
            print(f"    ✓ {os.path.basename(fpath)}: {len(df_c):,} rows")
        except Exception as e:
            print(f"    WARNING: {os.path.basename(fpath)}: {e}")
    if frames:
        cost_df = pd.concat(frames, ignore_index=True)
        col_map = {}
        for c in cost_df.columns:
            cl = c.lower()
            if "sett_date" in cl or "date" in cl:
                col_map[c] = "date_clean"
            elif "sett_period" in cl or "period" in cl:
                col_map[c] = "sett_period"
            elif "constraint" in cl:
                col_map[c] = "constraint_cost_gbp"
        cost_df.rename(columns=col_map, inplace=True)
        if "date_clean" in cost_df.columns:
            cost_df["date_clean"] = pd.to_datetime(cost_df["date_clean"], errors="coerce").dt.date
            cost_df["sett_period"] = pd.to_numeric(cost_df["sett_period"], errors="coerce").astype("Int64")
        print(f"    Total cost rows: {len(cost_df):,}")
else:
    print("\n  ℹ  No balancing cost files found")


# ── 4e. CfD data (daily, project-level context) ─────────────────────────────

cfd_path = glob.glob(os.path.join(RAW_DIR, "actual_cfd_generation*.csv"))
cfd_df = None
cfd_region = None

if cfd_path:
    print(f"\n  Loading CfD data …")
    try:
        cfd_raw = pd.read_csv(cfd_path[0], low_memory=False)
        cfd_raw.columns = [c.strip() for c in cfd_raw.columns]

        # Find date column
        date_col = next((c for c in cfd_raw.columns if "date" in c.lower()), None)
        if date_col:
            cfd_raw["date_clean"] = pd.to_datetime(cfd_raw[date_col], errors="coerce").dt.date

        # Try to find region-relevant CfD units
        name_col = next((c for c in cfd_raw.columns if "name" in c.lower()
                         and "cfd" in c.lower()), None)
        if name_col is None:
            name_col = next((c for c in cfd_raw.columns if "name" in c.lower()), None)

        cfd_df = cfd_raw
        print(f"    ✓ {len(cfd_raw):,} rows loaded")
        print(f"    Columns: {list(cfd_raw.columns)}")

        if name_col and REGION:
            region_match = cfd_raw[cfd_raw[name_col].astype(str).str.contains(
                REGION, case=False, na=False)]
            if len(region_match):
                cfd_region = region_match
                units = region_match[name_col].unique()
                print(f"    Region matches: {list(units)}")
                # Find payment columns
                pay_col = next((c for c in cfd_raw.columns
                                if "payment" in c.lower() and "gbp" in c.lower()), None)
                gen_col = next((c for c in cfd_raw.columns
                                if "generation" in c.lower() and "mwh" in c.lower()), None)
                if pay_col and gen_col:
                    total_pay = pd.to_numeric(region_match[pay_col], errors="coerce").sum()
                    total_gen = pd.to_numeric(region_match[gen_col], errors="coerce").sum()
                    print(f"    CfD payments: £{total_pay/1e6:,.1f}m")
                    print(f"    CfD generation: {total_gen/1e6:,.2f} TWh")
            else:
                print(f"    No CfD matches for '{REGION}'")
    except Exception as e:
        print(f"    WARNING: Could not parse CfD data: {e}")
else:
    print("\n  ℹ  No CfD data found")


# ══════════════════════════════════════════════════════════════════════════════
# 5. COMPUTE FINANCIAL VALUES
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"FINANCIAL ANALYSIS — {REGION_LABEL}")
print(f"{'='*70}")

# Merge price onto region settlement periods
region_sp["date_only"] = pd.to_datetime(region_sp["Date"]).dt.date
region_sp["sett_period_int"] = region_sp["Settlement_Period"].astype(int)

if price_series is not None:
    # Join IMRP prices
    region_sp["price_gbp_mwh"] = region_sp.apply(
        lambda r: price_series.get((r["date_only"], r["sett_period_int"]), np.nan),
        axis=1
    )
    matched_pct = region_sp["price_gbp_mwh"].notna().mean() * 100
    print(f"\n  Price match rate: {matched_pct:.0f}% of periods have IMRP prices")
    # Fill unmatched with reference
    region_sp["price_gbp_mwh"] = region_sp["price_gbp_mwh"].fillna(REFERENCE_PRICE_GBP_MWH)
    pricing_label = f"IMRP + reference fallback (£{REFERENCE_PRICE_GBP_MWH}/MWh)"
else:
    region_sp["price_gbp_mwh"] = REFERENCE_PRICE_GBP_MWH
    pricing_label = f"Reference price (£{REFERENCE_PRICE_GBP_MWH}/MWh)"
    print(f"\n  Using reference price: £{REFERENCE_PRICE_GBP_MWH}/MWh")

# Financial value of curtailed energy
region_sp["curtailment_value_gbp"] = region_sp["curtailment_MWh"] * region_sp["price_gbp_mwh"]

# DC absorption potential (simplified from 03/04)
avail_flex_mw = DC_FLEET_MW * DC_HEADROOM * DC_FLEX_FRAC
region_sp["absorbed_MW"]  = np.minimum(region_sp["curtailment_MW"].values, avail_flex_mw)
region_sp["absorbed_MWh"] = region_sp["absorbed_MW"] * 0.5
region_sp["absorbed_value_gbp"] = region_sp["absorbed_MWh"] * region_sp["price_gbp_mwh"]

# Totals
total_curt_gwh   = region_sp["curtailment_MWh"].sum() / 1000
total_curt_value  = region_sp["curtailment_value_gbp"].sum()
total_abs_gwh    = region_sp["absorbed_MWh"].sum() / 1000
total_abs_value   = region_sp["absorbed_value_gbp"].sum()
mean_price       = region_sp["price_gbp_mwh"].mean()

print(f"\n  Pricing: {pricing_label}")
print(f"  Mean price: £{mean_price:.1f}/MWh")
print(f"\n  {REGION_LABEL} curtailment:")
print(f"    Energy: {total_curt_gwh:,.1f} GWh")
print(f"    Value:  £{total_curt_value/1e6:,.0f}m")
print(f"\n  DC absorption potential ({DC_FLEET_MW} MW fleet, "
      f"{DC_FLEX_FRAC:.0%} flex, {DC_HEADROOM:.0%} headroom):")
print(f"    Energy: {total_abs_gwh:,.1f} GWh ({total_abs_gwh/total_curt_gwh*100:.1f}%)")
print(f"    Value:  £{total_abs_value/1e6:,.0f}m")


# ══════════════════════════════════════════════════════════════════════════════
# 6. SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════════

# Region curtailment per-period
region_sp.to_csv(os.path.join(CSV_DIR, "05_region_curtailment.csv"), index=False)

# Summary table
monthly = region_sp.groupby("Month").agg(
    curtailment_GWh=("curtailment_MWh", lambda x: x.sum() / 1000),
    curtailment_value_mGBP=("curtailment_value_gbp", lambda x: x.sum() / 1e6),
    absorbed_GWh=("absorbed_MWh", lambda x: x.sum() / 1000),
    absorbed_value_mGBP=("absorbed_value_gbp", lambda x: x.sum() / 1e6),
    mean_price_gbp_mwh=("price_gbp_mwh", "mean"),
    n_periods=("curtailment_MWh", "count"),
).reset_index()
monthly.to_csv(os.path.join(CSV_DIR, "05_financial_summary.csv"), index=False)

# Per-farm breakdown
farm_summary = (curt_region
    .groupby(["Generator_Name", "Generator_Full_Name"])
    .agg(
        events=("Curtailment_MWh", "count"),
        total_GWh=("Curtailment_MWh", lambda x: x.sum() / 1000),
    )
    .reset_index()
    .sort_values("total_GWh", ascending=False)
)
farm_summary.to_csv(os.path.join(CSV_DIR, "05_farm_breakdown.csv"), index=False)

print(f"\n  ✓ Saved 05_region_curtailment.csv")
print(f"  ✓ Saved 05_financial_summary.csv")
print(f"  ✓ Saved 05_farm_breakdown.csv")


# ══════════════════════════════════════════════════════════════════════════════
# 7. CHARTS
# ══════════════════════════════════════════════════════════════════════════════

print(f"\nGenerating charts …")

month_order = pd.period_range("2024-04", "2026-03", freq="M")


# ── 5a. Region vs System curtailment share ────────────────────────────────────

system_monthly = (curt_system
    .groupby(curt_system["Date"].dt.to_period("M"))["Curtailment_MWh"]
    .sum() / 1000)
region_monthly_gwh = (region_sp
    .groupby("Month")["curtailment_MWh"]
    .sum() / 1000)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: stacked bar
x = np.arange(len(month_order))
sys_vals   = system_monthly.reindex(month_order, fill_value=0).values
reg_vals   = region_monthly_gwh.reindex(month_order, fill_value=0).values
other_vals = sys_vals - reg_vals

ax1.bar(x, other_vals, color=COLORS["grey_light"], label="Other farms")
ax1.bar(x, reg_vals, bottom=other_vals, color=COLORS["red"], label=REGION_LABEL)
ax1.set_xticks(x)
ax1.set_xticklabels([m.strftime("%b\n%Y") if m.month in [1, 4, 7, 10]
                      else m.strftime("%b") for m in month_order], fontsize=7)
ax1.set_ylabel("Curtailed Energy (GWh)")
ax1.set_title(f"Monthly Curtailment: {REGION_LABEL} vs Rest of System")
ax1.legend()

# Right: share %
with np.errstate(invalid="ignore"):
    share = np.where(sys_vals > 0, reg_vals / sys_vals * 100, 0)
share = np.nan_to_num(share, 0.0)
ax2.bar(x, share, color=COLORS["red"], alpha=0.85)
mean_share = np.mean(share[share > 0]) if np.any(share > 0) else 0
ax2.axhline(y=mean_share, color="black", linestyle=":",
            label=f"Mean: {mean_share:.0f}%")
ax2.set_xticks(x)
ax2.set_xticklabels([m.strftime("%b\n%Y") if m.month in [1, 4, 7, 10]
                      else m.strftime("%b") for m in month_order], fontsize=7)
ax2.set_ylabel(f"{REGION_LABEL} Share (%)")
ax2.set_title(f"{REGION_LABEL}'s Share of Total UK Curtailment")
ax2.legend()

save_fig(fig, os.path.join(CHART_DIR, "05_region_vs_system.png"))


# ── 5b. Monthly financial impact ─────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(13, 5))
x = np.arange(len(month_order))

curt_val_monthly = (region_sp
    .groupby("Month")["curtailment_value_gbp"]
    .sum().reindex(month_order, fill_value=0) / 1e6)
abs_val_monthly = (region_sp
    .groupby("Month")["absorbed_value_gbp"]
    .sum().reindex(month_order, fill_value=0) / 1e6)

ax.bar(x - 0.2, curt_val_monthly.values, width=0.4,
       color=COLORS["red"], alpha=0.85, label=f"Curtailment cost (£m)")
ax.bar(x + 0.2, abs_val_monthly.values, width=0.4,
       color=COLORS["green"], alpha=0.85,
       label=f"DC absorption value (£m)\n({DC_FLEET_MW} MW fleet)")
ax.set_xticks(x)
ax.set_xticklabels([m.strftime("%b\n%Y") if m.month in [1, 4, 7, 10]
                     else m.strftime("%b") for m in month_order], fontsize=7)
ax.set_ylabel("Value (£ millions)")
ax.set_title(f"Monthly Financial Impact of {REGION_LABEL} Curtailment\n"
             f"Pricing: {pricing_label}")
ax.legend(fontsize=8)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"£{v:,.0f}m"))
save_fig(fig, os.path.join(CHART_DIR, "05_monthly_financial.png"))


# ── 5c. Hourly financial profile ─────────────────────────────────────────────

hourly_value = (region_sp.groupby("HourInt")
    .agg(
        curt_gwh=("curtailment_MWh", lambda x: x.sum() / 1000),
        curt_value_m=("curtailment_value_gbp", lambda x: x.sum() / 1e6),
        abs_value_m=("absorbed_value_gbp", lambda x: x.sum() / 1e6),
    )
    .reindex(range(24), fill_value=0)
)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

ax1.bar(range(24), hourly_value["curt_gwh"],
        color=COLORS["blue"], alpha=0.85)
ax1.set_ylabel("Curtailed Energy (GWh)")
ax1.set_title(f"{REGION_LABEL}: Curtailment by Hour of Day")

ax2.bar(range(24), hourly_value["curt_value_m"],
        color=COLORS["red"], alpha=0.7, label="Curtailment cost")
ax2.bar(range(24), hourly_value["abs_value_m"],
        color=COLORS["green"], alpha=0.85, label="DC absorption value")
ax2.set_ylabel("Value (£m)")
ax2.set_xlabel("Hour of Day")
ax2.set_title("Financial Value by Hour of Day")
hour_labels(ax2)
ax2.legend()

save_fig(fig, os.path.join(CHART_DIR, "05_hourly_financial.png"))


# ── 5d. Farm-level breakdown (within region) ─────────────────────────────────

fig, ax = plt.subplots(figsize=(10, max(4, len(farm_summary) * 0.5)))
ax.barh(range(len(farm_summary)), farm_summary["total_GWh"],
        color=COLORS["blue"])
ax.set_yticks(range(len(farm_summary)))
ax.set_yticklabels(farm_summary["Generator_Full_Name"])
ax.invert_yaxis()
ax.set_xlabel("Total Curtailed Energy (GWh)")
ax.set_title(f"{REGION_LABEL}: Curtailment by Individual Unit")
# Add value labels
for i, v in enumerate(farm_summary["total_GWh"]):
    ax.text(v + farm_summary["total_GWh"].max() * 0.01, i,
            f"{v:,.0f} GWh", va="center", fontsize=9)
save_fig(fig, os.path.join(CHART_DIR, "05_farm_breakdown.png"))


# ── 5e. Daily timeline (region) ──────────────────────────────────────────────

daily_reg = (region_sp
    .groupby("Date")
    .agg(
        curt_gwh=("curtailment_MWh", lambda x: x.sum() / 1000),
        value_m=("curtailment_value_gbp", lambda x: x.sum() / 1e6),
    )
    .reset_index()
)

fig, ax1 = plt.subplots(figsize=(14, 4))
ax1.fill_between(daily_reg["Date"], daily_reg["curt_gwh"],
                 alpha=0.3, color=COLORS["blue"])
ax1.plot(daily_reg["Date"], daily_reg["curt_gwh"],
         linewidth=0.5, color=COLORS["blue"], label="Curtailed GWh")
ax1.set_ylabel("Energy (GWh/day)")
ax1.set_title(f"Daily {REGION_LABEL} Curtailment")
ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %Y"))
ax2 = ax1.twinx()
ax2.plot(daily_reg["Date"], daily_reg["value_m"],
         linewidth=0.8, color=COLORS["red"], alpha=0.6, label="Value £m")
ax2.set_ylabel("Daily value (£m)", color=COLORS["red"])
ax2.tick_params(axis="y", labelcolor=COLORS["red"])
ax2.spines["right"].set_visible(True)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
save_fig(fig, os.path.join(CHART_DIR, "05_daily_timeline.png"))


# ── 5f. DC fleet sizing sensitivity (£ value) ────────────────────────────────

fleet_sizes = [100, 200, 500, 1000, 2000, 3000]
results = []
for fs in fleet_sizes:
    avail = fs * DC_HEADROOM * DC_FLEX_FRAC
    abs_mw  = np.minimum(region_sp["curtailment_MW"].values, avail)
    abs_mwh = abs_mw * 0.5
    abs_val = abs_mwh * region_sp["price_gbp_mwh"].values
    results.append({
        "fleet_mw": fs,
        "absorbed_gwh": abs_mwh.sum() / 1000,
        "value_m_gbp": abs_val.sum() / 1e6,
    })
sizing = pd.DataFrame(results)

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(sizing["fleet_mw"], sizing["absorbed_gwh"],
         color=COLORS["blue"], linewidth=2.5, marker="o", label="Absorbed GWh")
ax1.set_xlabel("DC Fleet Capacity (MW)")
ax1.set_ylabel("Absorbed Energy (GWh)", color=COLORS["blue"])
ax1.tick_params(axis="y", labelcolor=COLORS["blue"])

ax2 = ax1.twinx()
ax2.plot(sizing["fleet_mw"], sizing["value_m_gbp"],
         color=COLORS["green"], linewidth=2.5, marker="s", label="Value £m")
ax2.set_ylabel("Absorption Value (£m)", color=COLORS["green"])
ax2.tick_params(axis="y", labelcolor=COLORS["green"])
ax2.spines["right"].set_visible(True)

ax1.set_title(f"DC Fleet Sizing vs {REGION_LABEL} Curtailment Absorption Value\n"
              f"({DC_FLEX_FRAC:.0%} flex, {DC_HEADROOM:.0%} headroom, "
              f"£{REFERENCE_PRICE_GBP_MWH}/MWh ref)")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

for _, row in sizing.iterrows():
    ax2.annotate(f"£{row['value_m_gbp']:,.0f}m",
                 xy=(row["fleet_mw"], row["value_m_gbp"]),
                 textcoords="offset points", xytext=(0, 10),
                 ha="center", fontsize=8, color=COLORS["green"])

save_fig(fig, os.path.join(CHART_DIR, "05_fleet_sizing_value.png"))


# ── 5g. CfD context (if available) ───────────────────────────────────────────

if cfd_region is not None:
    pay_col = next((c for c in cfd_region.columns
                    if "payment" in c.lower() and "gbp" in c.lower()), None)
    gen_col = next((c for c in cfd_region.columns
                    if "generation" in c.lower() and "mwh" in c.lower()), None)

    if pay_col and gen_col and "date_clean" in cfd_region.columns:
        cfd_monthly = cfd_region.copy()
        cfd_monthly[pay_col] = pd.to_numeric(cfd_monthly[pay_col], errors="coerce")
        cfd_monthly[gen_col] = pd.to_numeric(cfd_monthly[gen_col], errors="coerce")
        cfd_monthly["date_clean"] = pd.to_datetime(cfd_monthly["date_clean"])
        cfd_monthly["Month"] = cfd_monthly["date_clean"].dt.to_period("M")

        cfd_agg = cfd_monthly.groupby("Month").agg(
            cfd_payments_m=pd.NamedAgg(column=pay_col, aggfunc=lambda x: x.sum() / 1e6),
            cfd_gen_gwh=pd.NamedAgg(column=gen_col, aggfunc=lambda x: x.sum() / 1000),
        )

        fig, ax = plt.subplots(figsize=(13, 5))
        x = np.arange(len(month_order))
        cfd_vals = cfd_agg.reindex(month_order).fillna(0)

        ax.bar(x - 0.2, curt_val_monthly.values, width=0.4,
               color=COLORS["red"], alpha=0.85, label="Curtailment cost (£m)")
        ax.bar(x + 0.2, cfd_vals["cfd_payments_m"].values, width=0.4,
               color=COLORS["purple"], alpha=0.85, label="CfD payments (£m)")
        ax.set_xticks(x)
        ax.set_xticklabels([m.strftime("%b\n%Y") if m.month in [1, 4, 7, 10]
                             else m.strftime("%b") for m in month_order], fontsize=7)
        ax.set_ylabel("£ millions")
        ax.set_title(f"{REGION_LABEL}: Curtailment Cost vs CfD Payments")
        ax.legend()
        save_fig(fig, os.path.join(CHART_DIR, "05_cfd_context.png"))


# ══════════════════════════════════════════════════════════════════════════════
# 8. FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"SUMMARY — {REGION_LABEL}")
print(f"{'='*70}")
print(f"""
  Region: {REGION_LABEL}
  Units:  {len(region_farms)} ({', '.join(region_bmus)})
  Period: {region_sp['Date'].min():%b %Y} – {region_sp['Date'].max():%b %Y}

  Curtailment:
    Energy:           {total_curt_gwh:>10,.1f} GWh
    System share:     {total_curt_gwh/system_total_gwh*100:>10.1f} %
    Financial value:  £{total_curt_value/1e6:>9,.0f} m  ({pricing_label})

  DC absorption potential ({DC_FLEET_MW} MW fleet):
    Energy absorbed:  {total_abs_gwh:>10,.1f} GWh ({total_abs_gwh/total_curt_gwh*100:.1f}%)
    Value recovered:  £{total_abs_value/1e6:>9,.0f} m

  Peak period: {region_sp['curtailment_MW'].max():,.0f} MW
  Mean period: {region_sp['curtailment_MW'].mean():,.0f} MW
""")

if pricing_source == "reference":
    print(f"  NOTE: Using assumed reference price £{REFERENCE_PRICE_GBP_MWH}/MWh.")
    print(f"  Add imrp_actuals.csv to data/raw/ for actual settlement prices.")

print(f"\n  Datasets loaded: BOA curtailment"
      f"{', IMRP prices' if pricing_source == 'IMRP' else ''}"
      f"{', BSAD' if bsad_df is not None else ''}"
      f"{', Balancing volumes' if vol_df is not None else ''}"
      f"{', Balancing costs' if cost_df is not None else ''}"
      f"{', CfD' if cfd_df is not None else ''}")
print(f"\n  ✓ Charts: {CHART_DIR}/05_*.png")
print(f"  ✓ CSVs:   {CSV_DIR}/05_*.csv")
print("  Done.\n")
