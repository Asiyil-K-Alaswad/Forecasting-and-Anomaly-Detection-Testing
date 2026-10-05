"""Correlation analysis, feature selection, distributions, outlier screening (Sec. III-C..E)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import config as C
from . import features as F
from .data import LABELS
from .plots import save, SERIES, INK_2, MUTED, CRITICAL, DIVERGING

# ~32 candidate quantities, as in the paper's Fig. 4
CANDIDATES = [
    "dc_power_kw", "dc_power_max_kw", "dc_power_min_kw", "dc_current_a", "dc_current_max_a",
    "dc_current_min_a", "energy_in_kwh", "dc_voltage_v", "conv_ratio", "load_swing",
    "mains_kwh", "mains_outage_h", "dg_kwh", "dg_h", "dg_starts", "dg_fuel_l", "solar_kwh",
    "batt_discharge_h", "llvd_h", "batt_temp_c", "batt_temp_max_c", "indoor_temp_c", "indoor_hum",
    "ran_energy_kwh", "lte_energy_kwh", "nr_energy_kwh", "traffic_gb", "dc_ran_ratio",
    "amb_temp_c", "amb_temp_max_c", "amb_humidity", "amb_wind_kph", "amb_pressure_mb",
]


def correlations(panel):
    x = panel[CANDIDATES]
    within = (x - x.groupby(panel["site"]).transform("mean")).corr()
    pooled = x.corr()
    within.to_csv(C.TAB_DIR / "correlation_within_site.csv")
    pooled.to_csv(C.TAB_DIR / "correlation_pooled.csv")
    return within, pooled


def plot_heatmap(corr, name, title):
    lab = [LABELS[c] for c in corr.columns]
    fig, ax = plt.subplots(figsize=(11.5, 10))
    im = ax.imshow(corr.values, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(lab)), lab, rotation=90, fontsize=7.5)
    ax.set_yticks(range(len(lab)), lab, fontsize=7.5)
    ax.grid(False)
    for i in range(len(lab)):
        for j in range(len(lab)):
            v = corr.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}".replace("0.", ".").replace("-.", "−."), ha="center",
                        va="center", fontsize=4.6, color="white" if abs(v) > 0.6 else INK_2)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.outline.set_visible(False)
    cb.set_label("Pearson r", color=INK_2)
    ax.set_title(title, loc="left", pad=10)
    return save(fig, name)


def table_one(within):
    r = lambda a, b: within.loc[a, b]  # noqa: E731
    rows = [
        ("DC load power vs current (avg/max/min)",
         f"P vs I r={r('dc_power_kw','dc_current_a'):.2f}; bus voltage is near-constant so P = V·I",
         "Keep Avg DC Load Power as the target; drop currents (redundant). DC Load Energy = 24·P exactly → drop"),
        ("Total energy input vs DC load",
         f"r={r('energy_in_kwh','dc_power_kw'):.2f}; diverges from output on battery charge/discharge days",
         "Not a forecasting input; used as the denominator of the conversion ratio (health indicator)"),
        ("RAN board energy vs LTE / NR energy",
         f"RAN vs LTE r={r('ran_energy_kwh','lte_energy_kwh'):.2f}, vs NR r={r('ran_energy_kwh','nr_energy_kwh'):.2f}",
         "Keep the aggregate RAN Board Energy (independent NMS meter, r="
         f"{r('dc_power_kw','ran_energy_kwh'):.2f} with load); drop LTE/NR split"),
        ("Traffic vs load",
         f"r={r('traffic_gb','dc_power_kw'):.2f} with load, {r('traffic_gb','ran_energy_kwh'):.2f} with RAN energy",
         "Retain Traffic as the demand driver"),
        ("Ambient temperature (avg/max), pressure, UV",
         f"avg vs max r={r('amb_temp_c','amb_temp_max_c'):.2f}; avg temp vs load r={r('amb_temp_c','dc_power_kw'):.2f}"
         f"; pressure vs temp r={r('amb_pressure_mb','amb_temp_c'):.2f}",
         "Keep Avg Ambient Temp (cooling-load driver); drop max/pressure (seasonal proxies)"),
        ("Ambient humidity",
         f"r={r('amb_humidity','dc_power_kw'):.2f} with load, {r('amb_humidity','amb_temp_c'):.2f} with temp",
         "Retain: partly independent weather signal"),
        ("Battery vs indoor temperature",
         f"r={r('batt_temp_c','indoor_temp_c'):.2f}; indoor sensors cover only ~59 % of site-days",
         "Keep Avg Battery Temp (thermal-stress indicator); drop indoor temp/humidity"),
        ("Battery discharge vs mains outage / DG duration",
         f"vs DG hours r={r('batt_discharge_h','dg_h'):.2f}, vs mains outage r={r('batt_discharge_h','mains_outage_h'):.2f}",
         "Keep Battery Discharge (valid for every power configuration); outage, DG, LLVD kept as root-cause indicators"),
        ("DC bus voltage",
         f"r={r('dc_voltage_v','batt_discharge_h'):.2f} with discharge, {r('dc_voltage_v','dc_power_kw'):.2f} with load",
         "Retain: voltage-quality indicator, largely independent of load"),
        ("Conversion ratio, load swing, DC/RAN ratio",
         f"max |r| with other selected features: conv {within.loc['conv_ratio', C.SELECTED_FEATURES].drop('conv_ratio').abs().max():.2f}, "
         f"swing {within.loc['load_swing', C.SELECTED_FEATURES].drop('load_swing').abs().max():.2f}, "
         f"DC/RAN {within.loc['dc_ran_ratio', C.SELECTED_FEATURES].drop('dc_ran_ratio').abs().max():.2f}",
         "Retain all three for unique health characterisation (efficiency, load distortion, meter consistency)"),
    ]
    t = pd.DataFrame(rows, columns=["Feature group", "Observation (within-site r)", "Recommendation"])
    t.to_csv(C.TAB_DIR / "table1_feature_selection.csv", index=False)
    return t


def plot_histograms(panel):
    feats = C.SELECTED_FEATURES
    fig, axes = plt.subplots(4, 3, figsize=(11, 11))
    for ax, c in zip(axes.flat, feats):
        x = panel[c].dropna().values
        lo, hi = np.percentile(x, [0.5, 99.5])
        x = x[(x >= lo) & (x <= hi)]
        ax.hist(x, bins=50, density=True, color=SERIES[0], alpha=0.55, edgecolor="none")
        grid, pdf, _, _ = F.binned_kde(x)
        ax.plot(grid, pdf, color=SERIES[1], lw=1.6)
        ax.set_xlim(lo, hi)
        ax.set_title(LABELS[c], loc="left", fontsize=9)
        ax.set_ylabel("Density", fontsize=8)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(0.02, 0.6, "Bars: histogram (central 99 %)\nLine: Gaussian KDE\n\n"
                       "Rows 1–2: forecasting inputs\nRows 2–4: health indicators\n(autoencoder inputs)",
                       color=INK_2, fontsize=9, va="top", transform=axes.flat[-1].transAxes)
    fig.suptitle("Distributions of the 11 selected features (all modelled site-days)", x=0.01,
                 ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return save(fig, "fig05_histograms.png")


def plot_boxplots(z, flags, split, panel):
    feats = C.SELECTED_FEATURES
    fig, axes = plt.subplots(1, len(feats), figsize=(13, 4.2), sharey=True)
    for ax, c in zip(axes, feats):
        v = z["z_" + c].dropna().values
        ax.boxplot(v, whis=C.IQR_K, widths=0.55, showfliers=True, patch_artist=True,
                   boxprops=dict(facecolor="#cde2fb", edgecolor=SERIES[0], linewidth=1),
                   medianprops=dict(color=SERIES[0], linewidth=1.6),
                   whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED),
                   flierprops=dict(marker="o", markersize=1.5, markerfacecolor=SERIES[1],
                                   markeredgecolor="none", alpha=0.12))
        rate = flags[c][z["z_" + c].notna()].mean() if c in flags else np.nan
        tag = "not screened\n(exogenous)" if np.isnan(rate) else f"outliers {rate:.2%}"
        ax.set_xticks([1], [LABELS[c].replace(" (", "\n(") + "\n\n" + tag], fontsize=6.5, rotation=0)
        ax.set_ylim(-12, 12)
    axes[0].set_ylabel("Site-relative z (fleet-scaled)")
    fig.suptitle("Box plots of the selected features after site-relative scaling (whiskers 1.5 IQR; "
                 "outliers = box-plot ∧ KDE consensus rate)", x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return save(fig, "fig06_boxplots.png")


def plot_kde_screening(z, split, panel):
    train = (panel["date"] < split["test_start"]).values
    feats = F.SCREENED
    fig, axes = plt.subplots(3, 3, figsize=(11, 7.5))
    for ax, c in zip(axes.flat, feats):
        v = z["z_" + c][train].dropna().values
        grid, pdf, level = F.kde_low_density_level(v, C.KDE_TAIL_MASS)
        ax.fill_between(grid, pdf, color=SERIES[0], alpha=0.1, lw=0)
        ax.plot(grid, pdf, color=SERIES[0], lw=1.6)
        tail = pdf < level
        ax.fill_between(grid, pdf, where=tail, color=CRITICAL, alpha=0.6, lw=0)
        ax.set_yscale("log")
        ax.set_ylim(1e-5, pdf.max() * 2)
        ax.set_xlim(-15, 15)
        ax.set_title(LABELS[c], loc="left", fontsize=9)
    fig.suptitle("Pooled KDE of site-relative z (training period, log density) — red: low-density "
                 f"region holding {C.KDE_TAIL_MASS:.0%} of the mass", x=0.01, ha="left", fontsize=10,
                 fontweight="bold")
    fig.tight_layout()
    return save(fig, "fig06b_kde_screening.png")
