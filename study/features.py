"""Site-relative normalisation, outlier screening (KDE + box plot) and partitioning.

Normalisation ("site-relative, fleet-scaled"):
    multiplicative features  u = x / mean_site_train - 1
    additive health features u = x - median_site_train
    weather (shared)         u = x - mean_train
    z = u / sigma_f          sigma_f = std of u over the training period (1-99 % winsorised)

All statistics come from the training period only, so the test period never
leaks into scaling, screening fences, or thresholds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

MULTIPLICATIVE = ["dc_power_kw", "traffic_gb", "ran_energy_kwh"]
WEATHER = ["amb_temp_c", "amb_humidity"]
ADDITIVE = C.AE_FEATURES
SCREENED = MULTIPLICATIVE + ADDITIVE   # site telemetry screened for outliers (weather is exogenous)


def split_dates(panel: pd.DataFrame):
    dates = np.sort(panel["date"].unique())
    n_train = int(round(len(dates) * C.TRAIN_FRACTION))
    n_val = int(round(n_train * C.VAL_FRACTION_OF_TRAIN))
    return {
        "start": pd.Timestamp(dates[0]),
        "val_start": pd.Timestamp(dates[n_train - n_val]),
        "test_start": pd.Timestamp(dates[n_train]),
        "end": pd.Timestamp(dates[-1]),
    }


def select_sites(panel: pd.DataFrame, split) -> pd.Index:
    """Sites with enough observed load history in the training period and some test data."""
    obs = panel["dc_power_kw"].notna()
    tr = panel[obs & (panel["date"] < split["test_start"])].groupby("site").size()
    te = panel[obs & (panel["date"] >= split["test_start"])].groupby("site").size()
    keep = tr[tr >= C.MIN_VALID_DAYS * C.TRAIN_FRACTION].index.intersection(te[te >= 30].index)
    return keep


def interpolate_short_gaps(panel: pd.DataFrame, cols) -> pd.DataFrame:
    """Linear interpolation inside a site for gaps of <= MAX_INTERP_GAP days only."""
    panel = panel.sort_values(["site", "date"]).copy()
    for c in cols:
        s = panel[c]
        filled = panel.groupby("site")[c].transform(
            lambda x: x.interpolate(limit=C.MAX_INTERP_GAP, limit_area="inside"))
        # interpolate(limit=k) also partially fills longer gaps; undo that.
        isna = s.isna()
        run_id = (isna != isna.groupby(panel["site"]).shift()).cumsum()
        run_len = isna.groupby(run_id).transform("sum")
        panel[c] = filled.where(~isna | (run_len <= C.MAX_INTERP_GAP))
    return panel


def fit_normaliser(panel: pd.DataFrame, split):
    tr = panel[panel["date"] < split["test_start"]]
    stats = {"site_center": {}, "sigma": {}}
    g = tr.groupby("site")
    for c in MULTIPLICATIVE:
        stats["site_center"][c] = g[c].mean()
    for c in ADDITIVE:
        stats["site_center"][c] = g[c].median()
    for c in WEATHER:
        stats["site_center"][c] = tr[c].mean()
    for c in MULTIPLICATIVE + ADDITIVE + WEATHER:
        u = _center(tr, c, stats)
        lo, hi = np.nanpercentile(u, [1, 99])
        stats["sigma"][c] = float(np.nanstd(np.clip(u, lo, hi)))
    return stats


def _center(df, c, stats):
    ctr = stats["site_center"][c]
    if c in WEATHER:
        return df[c] - ctr
    m = df["site"].map(ctr)
    if c in MULTIPLICATIVE:
        return df[c] / m - 1
    return df[c] - m


def transform(df: pd.DataFrame, stats) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in MULTIPLICATIVE + ADDITIVE + WEATHER:
        out["z_" + c] = _center(df, c, stats) / stats["sigma"][c]
    return out


def inverse_power(z, site_mean, stats):
    """z-space DC power -> kW."""
    return (np.asarray(z) * stats["sigma"]["dc_power_kw"] + 1) * np.asarray(site_mean)


# --------------------------------------------------------------------------
# KDE helpers (binned Gaussian KDE: fast and exact enough for 1-D screening)
# --------------------------------------------------------------------------
def binned_kde(x, grid_size=4096, bw=None, pad=0.15):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    lo, hi = np.percentile(x, [0.01, 99.99])
    span = hi - lo if hi > lo else 1.0
    lo, hi = lo - pad * span, hi + pad * span
    if bw is None:  # Silverman's rule
        iqr = np.subtract(*np.percentile(x, [75, 25]))
        sd = min(np.std(x), iqr / 1.349) if iqr > 0 else np.std(x)
        bw = 0.9 * max(sd, 1e-6) * len(x) ** (-0.2)
    grid = np.linspace(lo, hi, grid_size)
    dx = grid[1] - grid[0]
    counts, _ = np.histogram(np.clip(x, lo, hi), bins=grid_size, range=(lo - dx / 2, hi + dx / 2))
    k_half = int(np.ceil(4 * bw / dx))
    kern = np.exp(-0.5 * (np.arange(-k_half, k_half + 1) * dx / bw) ** 2)
    pdf = np.convolve(counts, kern, mode="same")
    pdf /= pdf.sum() * dx
    cdf = np.cumsum(pdf) * dx
    return grid, pdf, cdf, bw


def kde_percentile(x, q):
    """q-th percentile of the KDE-smoothed distribution (paper: dynamic KDE threshold)."""
    grid, _, cdf, _ = binned_kde(x)
    return float(np.interp(q / 100.0, cdf, grid))


def kde_low_density_level(x, tail_mass):
    """Density level d* such that the region {pdf < d*} carries `tail_mass` probability."""
    grid, pdf, _, _ = binned_kde(x)
    dx = grid[1] - grid[0]
    order = np.argsort(pdf)
    mass = np.cumsum(pdf[order]) * dx
    level = pdf[order][np.searchsorted(mass, tail_mass)]
    return grid, pdf, level


# --------------------------------------------------------------------------
# Outlier screening and partitioning (paper Sec. III-E)
# --------------------------------------------------------------------------
def screen_outliers(df: pd.DataFrame, z: pd.DataFrame, split):
    """Consensus outlier = outside the site's Tukey fences AND in the pooled-KDE low-density tail."""
    train = (df["date"] < split["test_start"]).values
    flags = pd.DataFrame(index=df.index)
    info = []
    for c in SCREENED:
        zc = z["z_" + c]
        q = zc[train].groupby(df.loc[train, "site"]).quantile([0.25, 0.75]).unstack()
        q1, q3 = df["site"].map(q[0.25]), df["site"].map(q[0.75])
        iqr = q3 - q1
        box = (zc < q1 - C.IQR_K * iqr) | (zc > q3 + C.IQR_K * iqr)
        grid, pdf, level = kde_low_density_level(zc[train].dropna().values, C.KDE_TAIL_MASS)
        dens = np.interp(zc.values, grid, pdf, left=0, right=0)
        kde = pd.Series((dens < level) & zc.notna().values, index=df.index)
        flags[c] = (box & kde).fillna(False)
        info.append({"feature": c, "box_rate": float(box[zc.notna()].mean()),
                     "kde_rate": float(kde[zc.notna()].mean()),
                     "consensus_rate": float(flags[c][zc.notna()].mean()),
                     "kde_level": float(level)})
    return flags, pd.DataFrame(info)
