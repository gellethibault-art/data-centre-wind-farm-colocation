"""
01_curtailment_analysis.py — BOA Wind Curtailment Pattern Analysis
Group 10: Wind Curtailment and Data Centres

Analyses the NESO Bid-Offer Acceptance (BOA) data to understand:
  - When wind curtailment happens (time of day, month, season)
  - How much energy is curtailed per settlement period
  - Which wind farms are curtailed most
  - Power levels (MW) to set the scale for data centre comparison

Input:  data/raw/boa_data_2024_25.csv   (semicolon-separated, Apr 2024 – Mar 2025)
        data/raw/boa_data_2025_26.csv   (comma-separated, Apr 2025 – Mar 2026)
Output: data/processed/curtailment_per_period.csv
        data/processed/curtailment_processed.csv
        output/charts/01_*.png
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
for d in [PROC_DIR, CHART_DIR, CSV_DIR]:
    os.makedirs(d, exist_ok=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import apply_style, COLORS, save_fig, hour_labels
apply_style()

# ── config ────────────────────────────────────────────────────────────────────
# BOA volumes are in MWh per settlement period (half-hour).
# Verify against the NESO data portal export documentation.
BOA_VOLUME_UNIT = "MWh"
TZ = "Europe/London"

# Date range for the full analysis window
DATE_LABEL = "Apr 2024 – Mar 2026"

# ── 1. load & parse ───────────────────────────────────────────────────────────
BOA_PATH_1 = os.path.join(RAW_DIR, "boa_data_2024_25.csv")
BOA_PATH_2 = os.path.join(RAW_DIR, "boa_data_2025_26.csv")
for p in [BOA_PATH_1, BOA_PATH_2]:
    if not os.path.exists(p):
        sys.exit(f"ERROR: Cannot find {p}\nPlace the BOA CSV in data/raw/")

print("Loading BOA data …")
df_1 = pd.read_csv(BOA_PATH_1, sep=";")
df_2 = pd.read_csv(BOA_PATH_2, sep=",")

df_1["Date"] = pd.to_datetime(df_1["Date"], dayfirst=True)
df_2["Date"] = pd.to_datetime(df_2["Date"], format="ISO8601")

df = pd.concat([df_1, df_2], ignore_index=True)


# ── DST-safe settlement-period → datetime mapping ────────────────────────────

def add_sp_datetime(df):
    """Map (Date, Settlement_Period) → timezone-aware datetime, handling
    25/23-hour clock-change days correctly."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

    out = []
    for d, g in df.groupby(df["Date"].dt.date, sort=False):
        day_start = pd.Timestamp(d).tz_localize(TZ)
        day_end   = day_start + pd.Timedelta(days=1)
        starts    = pd.date_range(day_start, day_end, freq="30min", inclusive="left")
        n = len(starts)

        sp = g["Settlement_Period"].astype(int).to_numpy()
        dt = pd.Series(pd.NaT, index=g.index)

        ok = (sp >= 1) & (sp <= n)
        dt.loc[g.index[ok]] = [starts[i - 1].tz_localize(None) for i in sp[ok]]
        out.append(dt)

    df["Datetime"] = pd.concat(out).sort_index()
    df = df.dropna(subset=["Datetime"])
    df["Hour"]    = df["Datetime"].dt.hour + df["Datetime"].dt.minute / 60
    df["HourInt"] = df["Datetime"].dt.hour
    return df


df = add_sp_datetime(df)

# ── Unit conversion ───────────────────────────────────────────────────────────
if BOA_VOLUME_UNIT == "MWh":
    df["Curtailment_MWh"] = df["BOA_Volume"].abs()
    df["Curtailment_MW"]  = df["Curtailment_MWh"] / 0.5
elif BOA_VOLUME_UNIT == "MW":
    df["Curtailment_MW"]  = df["BOA_Volume"].abs()
    df["Curtailment_MWh"] = df["Curtailment_MW"] * 0.5
else:
    raise ValueError(f"BOA_VOLUME_UNIT must be 'MWh' or 'MW', got {BOA_VOLUME_UNIT}")

df["Month"]      = df["Date"].dt.to_period("M")
df["MonthLabel"] = df["Date"].dt.strftime("%b %Y")
df["DayOfWeek"]  = df["Date"].dt.day_name()

print(f"  {len(df):,} curtailment events loaded")
print(f"  Date range: {df['Date'].min():%d %b %Y} → {df['Date'].max():%d %b %Y}")
print(f"  Total curtailed: {df['Curtailment_MWh'].sum()/1000:,.0f} GWh")
print(f"  Unique wind farms: {df['Generator_Name'].nunique()}")


# ── 2. aggregate per settlement period ────────────────────────────────────────

sp = df.groupby(["Date", "Settlement_Period"]).agg(
    total_curtailment_MWh=("Curtailment_MWh", "sum"),
    num_farms_curtailed=("Generator_Name", "nunique"),
).reset_index()

sp = sp.merge(
    df[["Date", "Settlement_Period", "Datetime"]].drop_duplicates(),
    on=["Date", "Settlement_Period"], how="left",
)

sp["curtailment_MW"] = sp["total_curtailment_MWh"] / 0.5
sp["Hour"]    = sp["Datetime"].dt.hour + sp["Datetime"].dt.minute / 60
sp["HourInt"] = sp["Datetime"].dt.hour
sp["Month"]   = sp["Date"].dt.to_period("M")

total_half_hours = 365.25 * 2 * 48  # ~2 years
curt_pct = len(sp) / total_half_hours * 100

print(f"\n  Settlement periods with curtailment: {len(sp):,} / ~{total_half_hours:.0f} ({curt_pct:.0f}%)")
print(f"  Mean curtailment:   {sp['curtailment_MW'].mean():,.0f} MW")
print(f"  Median curtailment: {sp['curtailment_MW'].median():,.0f} MW")
print(f"  Max curtailment:    {sp['curtailment_MW'].max():,.0f} MW")


# ── 3. save processed data ───────────────────────────────────────────────────

df.to_csv(os.path.join(PROC_DIR, "curtailment_processed.csv"), index=False)
sp.to_csv(os.path.join(PROC_DIR, "curtailment_per_period.csv"), index=False)
print(f"\n  ✓ Saved curtailment_processed.csv")
print(f"  ✓ Saved curtailment_per_period.csv")


# ══════════════════════════════════════════════════════════════════════════════
# 4. CHARTS
# ══════════════════════════════════════════════════════════════════════════════

month_order = pd.period_range("2024-04", "2026-03", freq="M")

# ── 4a. Monthly curtailment (bar + event count) ──────────────────────────────

monthly = sp.groupby("Month")["total_curtailment_MWh"].agg(["sum", "count"]).reindex(month_order)

fig, ax1 = plt.subplots(figsize=(13, 5))
ax1.bar(range(len(month_order)), monthly["sum"] / 1000,
        color=COLORS["blue"], alpha=0.85, label="Curtailed energy (GWh)")
ax1.set_ylabel("Curtailed Energy (GWh)")
ax1.set_title(f"Monthly Wind Curtailment — UK Balancing Mechanism ({DATE_LABEL})")
ax2 = ax1.twinx()
ax2.plot(range(len(month_order)), monthly["count"],
         color=COLORS["red"], marker="o", linewidth=2, label="Curtailment events")
ax2.set_ylabel("Curtailment Events", color=COLORS["red"])
ax2.tick_params(axis="y", labelcolor=COLORS["red"])
ax2.spines["right"].set_visible(True)
ax1.set_xticks(range(len(month_order)))
ax1.set_xticklabels([m.strftime("%b\n%Y") if m.month in [1, 4, 7, 10] else m.strftime("%b")
                      for m in month_order], fontsize=8)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
save_fig(fig, os.path.join(CHART_DIR, "01_monthly_curtailment.png"))


# ── 4b. Hourly curtailment (time-of-day bar chart) ──────────────────────────

hourly = sp.groupby("HourInt")["total_curtailment_MWh"].sum().reindex(range(24), fill_value=0)

fig, ax = plt.subplots(figsize=(12, 5))
bar_colors = [COLORS["night"] if h < 6 or h >= 22
              else COLORS["shoulder"] if h < 9 or h >= 17
              else COLORS["day"] for h in range(24)]
ax.bar(range(24), hourly.values / 1000, color=bar_colors, edgecolor="white", width=0.85)
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Total Curtailed Energy (GWh)")
ax.set_title(f"Wind Curtailment by Hour of Day ({DATE_LABEL})")
hour_labels(ax)
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(facecolor=COLORS["night"],    label="Night (22:00–05:59)"),
    Patch(facecolor=COLORS["shoulder"], label="Morning / Evening"),
    Patch(facecolor=COLORS["day"],      label="Daytime (09:00–16:59)"),
], loc="upper right")
save_fig(fig, os.path.join(CHART_DIR, "01_hourly_curtailment.png"))


# ── 4c. Heatmap: hour × month ────────────────────────────────────────────────

pivot = sp.groupby(["Month", "HourInt"])["total_curtailment_MWh"].sum().reset_index()
heat  = pivot.pivot(index="HourInt", columns="Month", values="total_curtailment_MWh")
heat  = heat[[m for m in month_order if m in heat.columns]]

fig, ax = plt.subplots(figsize=(14, 7))
im = ax.imshow(heat.values / 1000, aspect="auto", cmap="YlOrRd", interpolation="nearest")
ax.set_yticks(range(24))
ax.set_yticklabels([f"{h:02d}:00" for h in range(24)])
ax.set_xticks(range(len(heat.columns)))
ax.set_xticklabels([m.strftime("%b %Y") for m in heat.columns], rotation=45, ha="right")
ax.set_ylabel("Hour of Day")
ax.set_title(f"Curtailment Intensity Heatmap — Hour × Month (GWh)")
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("Curtailed Energy (GWh)")
save_fig(fig, os.path.join(CHART_DIR, "01_heatmap_hour_month.png"))


# ── 4d. Daily timeline ───────────────────────────────────────────────────────

daily = sp.groupby("Date")["total_curtailment_MWh"].sum().reset_index()

fig, ax = plt.subplots(figsize=(14, 4))
ax.fill_between(daily["Date"], daily["total_curtailment_MWh"] / 1000,
                alpha=0.3, color=COLORS["blue"])
ax.plot(daily["Date"], daily["total_curtailment_MWh"] / 1000,
        linewidth=0.6, color=COLORS["blue"])
ax.set_ylabel("Curtailed Energy (GWh)")
ax.set_title(f"Daily Wind Curtailment ({DATE_LABEL})")
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %Y"))
save_fig(fig, os.path.join(CHART_DIR, "01_daily_timeline.png"))


# ── 4e. Top 15 curtailed wind farms ──────────────────────────────────────────

top = (df.groupby(["Generator_Name", "Generator_Full_Name"])["Curtailment_MWh"]
       .sum().reset_index().sort_values("Curtailment_MWh", ascending=False).head(15))

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(range(len(top)), top["Curtailment_MWh"] / 1000, color=COLORS["blue"])
ax.set_yticks(range(len(top)))
ax.set_yticklabels(top["Generator_Full_Name"])
ax.invert_yaxis()
ax.set_xlabel("Total Curtailed Energy (GWh)")
ax.set_title(f"Top 15 Most Curtailed Wind Farms ({DATE_LABEL})")
save_fig(fig, os.path.join(CHART_DIR, "01_top_farms.png"))


# ── 4f. Power-level distribution with DC thresholds ─────────────────────────

fig, ax = plt.subplots(figsize=(12, 5))
bins = np.arange(0, sp["curtailment_MW"].max() + 500, 250)
ax.hist(sp["curtailment_MW"], bins=bins, color=COLORS["blue"], alpha=0.75,
        edgecolor="white")
thresholds = [
    (100,  COLORS["green"],  "100 MW (single DC)"),
    (500,  COLORS["amber"],  "500 MW (DC cluster)"),
    (1000, COLORS["red"],    "1 GW (large portfolio)"),
    (2000, COLORS["purple"], "2 GW (future scenario)"),
]
for mw, col, lbl in thresholds:
    ax.axvline(x=mw, color=col, linestyle="--", linewidth=2, label=lbl)
ax.set_xlabel("System Curtailment Power (MW)")
ax.set_ylabel("Number of Settlement Periods")
ax.set_title("Distribution of Half-Hourly Curtailment Power Levels")
ax.legend(fontsize=9)
save_fig(fig, os.path.join(CHART_DIR, "01_power_distribution.png"))


# ── 4g. Lowest-three months zoom ─────────────────────────────────────────────

monthly_totals = daily.groupby(daily["Date"].dt.to_period("M"))["total_curtailment_MWh"].sum()
low3 = monthly_totals.nsmallest(3).index
daily["Month"] = daily["Date"].dt.to_period("M")
low_daily = daily[daily["Month"].isin(low3)]

fig, ax = plt.subplots(figsize=(14, 4))
for m in low3:
    dsub = low_daily[low_daily["Month"] == m]
    ax.plot(dsub["Date"], dsub["total_curtailment_MWh"] / 1000,
            label=m.strftime("%b %Y"), linewidth=1.5)
ax.set_ylabel("Curtailed Energy (GWh)")
ax.set_title("Daily Curtailment in Lowest-Three Months")
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%d %b"))
ax.legend()
save_fig(fig, os.path.join(CHART_DIR, "01_daily_timeline_low3.png"))

print(f"\n  ✓ All charts saved to {CHART_DIR}/01_*.png")
print("  Done.\n")
