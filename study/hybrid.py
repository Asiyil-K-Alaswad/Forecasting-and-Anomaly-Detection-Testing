"""Hybrid decision logic and root-cause analysis (paper Algorithm 1, Sec. IV-C)."""
from __future__ import annotations

import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from . import features as F
from .data import LABELS
from .plots import save, SERIES, INK_2, MUTED, CRITICAL, WARNING
from .prepare import Prepared

Z = 3.0  # indicator deviation (site-relative, fleet-scaled z) considered "elevated"

DIAGNOSES = [
    ("Supply interruption (mains outage / battery discharge)",
     lambda p: (p["mains_outage_h"] >= 1) | (p["z_batt_discharge_h"] >= Z)),
    ("Load disconnection (LLVD/BLVD operated)", lambda p: (p["llvd_h"] > 0) | (p["blvd_h"] > 0)),
    ("Battery thermal stress", lambda p: (p["z_batt_temp_c"] >= Z) | (p["batt_temp_max_c"] >= 45)),
    ("Rectifier conversion anomaly (DC out / AC in)", lambda p: p["z_conv_ratio"].abs() >= Z),
    ("DC bus voltage deviation", lambda p: p["z_dc_voltage_v"].abs() >= Z),
    ("Load distortion (high intra-day swing)", lambda p: p["z_load_swing"].abs() >= Z),
    ("DC vs RAN meter mismatch", lambda p: p["z_dc_ran_ratio"].abs() >= Z),
    ("Traffic-driven load change", lambda p: p["z_traffic_gb"].abs() >= Z),
    ("Hardware change (board power-on within ±3 days)", lambda p: p["hw_near"] > 0),
    ("DG abnormal runtime", lambda p: p["dg_abnormal_h"] > 0),
]


def forecast_residual(prep: Prepared, preds_kw, ref_kw, sc_p):
    """Signed day-ahead residual relative to the forecast window's own load level.

    res(d) = (P(d) - P_hat(d)) / L(d-1) / sc_p, where L is the 7-day reference level
    of the window the forecast was issued from and sc_p the fleet scale of
    window-relative load deviations: a local, fleet-comparable z-score.
    """
    actual = np.where(prep.observed, prep.to_grid("dc_power_kw"), np.nan)
    f = np.full_like(actual, np.nan)
    f[:, 1:] = preds_kw[:, :-1, 0]
    ref = np.full_like(actual, np.nan)
    ref[:, 1:] = ref_kw[:, :-1]
    return (actual - f) / ref / sc_p


def decisions(prep: Prepared, fc, err, tau_a, tau_f=None, tau_f_adj=None):
    """fc = (preds_kw, ref_kw, sc_p) from the forecasting stage."""
    S, T = err.shape
    tt = np.arange(T)[None, :]
    v, te = prep.idx["val"], prep.idx["test"]
    res = forecast_residual(prep, *fc)
    rf = np.abs(res)
    sel = (tt >= v) & (tt < te) & ~prep.fc_irr & np.isfinite(rf)
    if tau_f is None:
        tau_f = F.kde_percentile(rf[sel], C.THRESHOLD_PERCENTILE)
    err = np.where(prep.observed, err, np.nan)
    test = (tt >= te) & prep.observed
    sf, sa = test & np.isfinite(rf), test & np.isfinite(err)
    af = sf & (rf > tau_f)
    aa = sa & (err > tau_a)

    # Fleet extension: remove the common-mode component (median residual of all sites of the
    # same type on that day), which carries weather / calendar shocks shared by the whole fleet.
    groups = prep.static["site_type"].to_numpy()
    cm = np.full_like(res, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for g in np.unique(groups):
            m = groups == g
            cm[m] = np.nanmedian(res[m], axis=0)
    rf_adj = np.abs(res - cm)
    if tau_f_adj is None:
        sel_adj = (tt >= v) & (tt < te) & ~prep.fc_irr & np.isfinite(rf_adj)
        tau_f_adj = F.kde_percentile(rf_adj[sel_adj], C.THRESHOLD_PERCENTILE)
    af_adj = sf & (rf_adj > tau_f_adj)
    return {"res": res, "rf": rf, "err": err, "tau_f": tau_f, "tau_a": tau_a, "test": test,
            "scored_f": sf, "scored_a": sa, "A_f": af, "A_a": aa, "A_or": af | aa, "A_and": af & aa,
            "common_mode": cm, "rf_adj": rf_adj, "tau_f_adj": tau_f_adj, "A_f_adj": af_adj,
            "A_or_adj": af_adj | aa, "A_and_adj": af_adj & aa}


def agreement(dd):
    both = dd["scored_f"] & dd["scored_a"]
    f, a = dd["A_f"][both], dd["A_a"][both]
    n = both.sum()
    po = np.mean(f == a)
    pe = f.mean() * a.mean() + (1 - f.mean()) * (1 - a.mean())
    return {"days_scored_by_both": int(n),
            "P(A_a | A_f)": float(a[f].mean()), "P(A_a | not A_f)": float(a[~f].mean()),
            "lift": float(a[f].mean() / a[~f].mean()),
            "jaccard": float((f & a).sum() / max((f | a).sum(), 1)), "cohen_kappa": float((po - pe) / (1 - pe))}


@np.errstate(all="ignore")
def episodes(prep: Prepared, dd, diag):
    """Group consecutive OR-flagged days at a site into episodes and describe each."""
    warnings.simplefilter("ignore", RuntimeWarning)
    flag = dd["A_or"]
    rows = []
    for s in np.where(flag.any(1))[0]:
        f = flag[s].astype(int)
        edges = np.diff(np.concatenate([[0], f, [0]]))
        for a, b in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
            sl = slice(a, b)
            labels = sorted({lab for lab in diag.columns if diag[lab].to_numpy().reshape(flag.shape)[s, sl].any()})
            ff = prep.ae_ferr_share[s, sl]
            top = C.AE_FEATURES[int(np.nanargmax(np.nanmean(ff, 0)))] if np.isfinite(ff).any() else ""
            rows.append({
                "site": prep.sites[s], "power_config": prep.static.iloc[s]["power_config"],
                "site_type": prep.static.iloc[s]["site_type"],
                "start": prep.dates[a].date(), "end": prep.dates[b - 1].date(), "days": int(b - a),
                "forecast_flag_days": int(dd["A_f"][s, sl].sum()), "ae_flag_days": int(dd["A_a"][s, sl].sum()),
                "both_flag_days": int(dd["A_and"][s, sl].sum()),
                "max_abs_residual_z": float(np.nanmax(dd["rf"][s, sl])) if np.isfinite(dd["rf"][s, sl]).any() else np.nan,
                "max_recon_error": float(np.nanmax(dd["err"][s, sl])) if np.isfinite(dd["err"][s, sl]).any() else np.nan,
                "load_change_%": float(np.nanmean(dd["res"][s, sl]) * dd["sc_p"] * 100)
                if np.isfinite(dd["res"][s, sl]).any() else np.nan,
                "top_ae_feature": LABELS.get(top, ""),
                "diagnosis": "; ".join(labels) if labels else "Unexplained by indicators",
            })
    ep = pd.DataFrame(rows)
    ep["severity"] = ep["max_abs_residual_z"].fillna(0) / dd["tau_f"] + ep["max_recon_error"].fillna(0) / dd["tau_a"]
    return ep.sort_values("severity", ascending=False)


def diagnose(prep: Prepared, dd):
    p = prep.panel
    hw = p["hw_poweron_n"].to_numpy().reshape(len(prep.sites), len(prep.dates))
    near = np.zeros_like(hw)
    T = hw.shape[1]
    for k in range(-3, 4):  # events within +-3 days, no wrap-around at the calendar edges
        if k >= 0:
            near[:, k:] += hw[:, :T - k]
        else:
            near[:, :k] += hw[:, -k:]
    p = p.assign(hw_near=near.ravel())
    out = pd.DataFrame({name: rule(p).fillna(False).to_numpy() for name, rule in DIAGNOSES})
    # a forecast flag that disappears once the fleet-wide shift of the day is removed
    cmode = dd["A_f"] & (dd["rf_adj"] <= dd["tau_f_adj"])
    out["Fleet-wide common-mode shift (weather / calendar)"] = cmode.ravel()
    return out


def root_cause_tables(prep: Prepared, dd, diag):
    shape = dd["A_or"].shape
    groups = {"Forecast only": dd["A_f"] & ~dd["A_a"], "Autoencoder only": dd["A_a"] & ~dd["A_f"],
              "Both (AND)": dd["A_and"], "Any (OR)": dd["A_or"]}
    normal = dd["test"] & (dd["scored_f"] | dd["scored_a"]) & ~dd["A_or"]
    rows = []
    for name in diag.columns:
        d = diag[name].to_numpy().reshape(shape)
        r = {"diagnosis": name, "normal days": float(d[normal].mean())}
        for g, m in groups.items():
            r[g] = float(d[m].mean())
        rows.append(r)
    any_diag = diag.any(axis=1).to_numpy().reshape(shape)
    r = {"diagnosis": "None of the above (unexplained)", "normal days": float((~any_diag)[normal].mean())}
    for g, m in groups.items():
        r[g] = float((~any_diag)[m].mean())
    rows.append(r)
    t = pd.DataFrame(rows)
    t.loc[len(t)] = {"diagnosis": "site-days", "normal days": int(normal.sum()),
                     **{g: int(m.sum()) for g, m in groups.items()}}
    t.to_csv(C.TAB_DIR / "root_cause_rates.csv", index=False)

    # Indicator elevation: mean |z| on flagged vs normal days (paper Fig. 9 analogue)
    el = []
    for c in F.SCREENED:
        zc = np.abs(prep.panel["z_" + c].to_numpy().reshape(shape))
        el.append({"indicator": LABELS[c], "feature": c,
                   "normal_mean_abs_z": float(np.nanmean(zc[normal])),
                   "flagged_mean_abs_z": float(np.nanmean(zc[dd["A_or"]])),
                   "ae_flag_mean_abs_z": float(np.nanmean(zc[dd["A_a"]])),
                   "fc_flag_mean_abs_z": float(np.nanmean(zc[dd["A_f"]]))})
    el = pd.DataFrame(el)
    el["elevation_factor"] = el["flagged_mean_abs_z"] / el["normal_mean_abs_z"]
    el.to_csv(C.TAB_DIR / "indicator_elevation.csv", index=False)

    # AE feature contribution share on autoencoder-flagged days
    share = prep.ae_ferr_share[dd["A_a"]]
    contrib = pd.DataFrame({"feature": [LABELS[c] for c in C.AE_FEATURES],
                            "mean_error_share": np.nanmean(share, 0),
                            "top_contributor_share": [np.mean(np.nanargmax(np.nan_to_num(share, nan=-1), 1) == k)
                                                      for k in range(len(C.AE_FEATURES))]})
    contrib.to_csv(C.TAB_DIR / "ae_feature_contribution.csv", index=False)
    return t, el, contrib


def plot_root_cause(contrib, rc):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1, 1.5]})
    ax = axes[0]
    c = contrib.sort_values("mean_error_share")
    ax.barh(c["feature"], c["mean_error_share"] * 100, color=SERIES[0], height=0.55)
    for y, v in enumerate(c["mean_error_share"] * 100):
        ax.text(v + 0.8, y, f"{v:.0f}%", va="center", fontsize=8, color=INK_2)
    ax.set_xlabel("Mean share of reconstruction error (%)")
    ax.set_title("(a) Feature contribution on autoencoder-flagged days", loc="left")
    ax.grid(axis="y", visible=False)

    ax = axes[1]
    r = rc[~rc["diagnosis"].isin(["site-days"])].copy()
    r = r.sort_values("Any (OR)")
    y = np.arange(len(r))
    hgt = 0.38
    ax.barh(y + hgt / 2, r["Any (OR)"] * 100, height=hgt, color=SERIES[1], label="Flagged site-days (OR)")
    ax.barh(y - hgt / 2, r["normal days"] * 100, height=hgt, color=MUTED, label="Normal site-days")
    ax.set_yticks(y, r["diagnosis"], fontsize=8)
    ax.set_xlabel("Share of site-days with the indicator present (%)")
    ax.set_title("(b) Root-cause indicators: flagged vs normal days", loc="left")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return save(fig, "fig09_root_cause.png")


def plot_fleet_daily(prep: Prepared, dd):
    te = prep.idx["test"]
    d = prep.dates[te:]
    rows = {}
    for k, lab in [("A_f", "Forecast residual > τf"), ("A_a", "Reconstruction error > τa"),
                   ("A_and", "Both (AND)"), ("A_f_adj", "Fleet-adjusted forecast residual")]:
        denom = (dd["scored_f"] | dd["scored_a"])[:, te:].sum(0)
        rows[lab] = dd[k][:, te:].sum(0) / np.maximum(denom, 1) * 100
    fig, ax = plt.subplots(figsize=(12, 3.6))
    for i, (lab, v) in enumerate(rows.items()):
        ax.plot(d, v, color=SERIES[i], lw=1.6, label=lab)
    ax.set_ylabel("Sites flagged (%)")
    ax.set_title("Fleet view: share of scored sites flagged each day (test period)", loc="left")
    ax.legend(loc="upper right", ncol=2)
    pd.DataFrame(rows, index=d).to_csv(C.TAB_DIR / "fleet_daily_flag_share.csv")
    return save(fig, "fig10_fleet_daily_flags.png")


def plot_site_hybrid(prep: Prepared, dd, preds_kw, i):
    v = prep.idx["val"]
    d = prep.dates[v:]
    actual = np.where(prep.observed[i], prep.to_grid("dc_power_kw")[i], np.nan)[v:]
    f1 = np.full(len(prep.dates), np.nan)
    f1[1:] = preds_kw[i, :-1, 0]
    f1 = f1[v:]
    af, aa = dd["A_f"][i, v:], dd["A_a"][i, v:]
    fig, axes = plt.subplots(3, 1, figsize=(12, 7.5), sharex=True, gridspec_kw={"height_ratios": [2, 2, 0.8]})
    ax = axes[0]
    ax.plot(d, actual, color=SERIES[0], lw=1.6, label="Actual")
    ax.plot(d, f1, color=SERIES[1], lw=1.2, label="Day-ahead forecast")
    ax.scatter(d[af], actual[af], s=40, facecolor="none", edgecolor=CRITICAL, lw=1.4, zorder=3,
               label="Residual > τf")
    ax.set_ylabel("DC load (kW)")
    ax.legend(loc="upper left", ncol=3)
    ax.set_title(f"Hybrid decision timeline — site {prep.sites[i]} (validation + test period)", loc="left")
    ax = axes[1]
    e = dd["err"][i, v:]
    ax.plot(d, e, color=SERIES[0], lw=1.2, label="Reconstruction error")
    ax.axhline(dd["tau_a"], color=WARNING, lw=1.4, label="τa")
    ax.scatter(d[aa], e[aa], marker="x", s=30, color=CRITICAL, lw=1.4, zorder=3, label="Error > τa")
    ax.set_yscale("log")
    ax.set_ylabel("AE error (log)")
    ax.legend(loc="upper left", ncol=3)
    ax = axes[2]
    for k, (lab, m) in enumerate([("OR", af | aa), ("AND", af & aa)]):
        ax.scatter(d[m], np.full(m.sum(), k), marker="|", s=200, color=CRITICAL if k else SERIES[1], lw=2)
    ax.set_yticks([0, 1], ["OR", "AND"])
    ax.set_ylim(-0.6, 1.6)
    ax.grid(axis="y", visible=False)
    for a in axes:
        a.axvline(prep.dates[prep.idx["test"]], color=INK_2, lw=0.8)
    fig.tight_layout()
    return save(fig, "fig08c_hybrid_site.png")


def run(prep: Prepared, rep_site_i: int):
    preds = np.load(C.CACHE_DIR / "fc_preds_kw.npy")
    ref = np.load(C.CACHE_DIR / "fc_ref_kw.npy")
    sc_p = json.load(open(C.TAB_DIR / "forecast_meta.json"))["window_scaling"]["dc_power_kw"]
    err = np.load(C.CACHE_DIR / "ae_err.npy")
    ferr = np.load(C.CACHE_DIR / "ae_ferr.npy")
    tau_a = json.load(open(C.TAB_DIR / "table4_autoencoder_metrics.json"))["threshold"]
    prep.ae_ferr_share = ferr / np.nansum(ferr, axis=2, keepdims=True)
    dd = decisions(prep, (preds, ref, sc_p), err, tau_a)
    dd["sc_p"] = sc_p
    n_test = int(dd["test"].sum())
    summary = {
        "test_site_days_observed": n_test,
        "coverage_forecast": float(dd["scored_f"].sum() / n_test),
        "coverage_autoencoder": float(dd["scored_a"].sum() / n_test),
        "tau_f_abs_residual_z": float(dd["tau_f"]),
        "tau_f_relative_load_error_%": float(dd["tau_f"] * sc_p * 100),
        "tau_a": float(tau_a),
    }
    summary["tau_f_adj_abs_residual_z"] = float(dd["tau_f_adj"])
    for k in ["A_f", "A_a", "A_or", "A_and", "A_f_adj", "A_or_adj", "A_and_adj"]:
        summary[f"{k}_site_days"] = int(dd[k].sum())
        summary[f"{k}_rate"] = float(dd[k].sum() / (dd["scored_f"] | dd["scored_a"]).sum())
        summary[f"{k}_sites"] = int(dd[k].any(1).sum())
    summary.update(agreement(dd))
    summary["forecast_flags_explained_by_common_mode"] = float(
        (dd["A_f"] & (dd["rf_adj"] <= dd["tau_f_adj"])).sum() / max(dd["A_f"].sum(), 1))
    te = prep.idx["test"]
    summary["corr_daily_common_mode_vs_temp_change"] = float(pd.Series(
        np.nanmedian(dd["common_mode"][:, te:], 0)).corr(pd.Series(
            prep.panel.groupby("date")["amb_temp_c"].first().diff().to_numpy()[te:])))
    diag = diagnose(prep, dd)
    ep = episodes(prep, dd, diag)
    summary["episodes_or"] = int(len(ep))
    summary["episode_median_days"] = float(ep["days"].median())
    summary["episodes_with_both_detectors"] = int((ep["both_flag_days"] > 0).sum())
    print(json.dumps(summary, indent=2))
    json.dump(summary, open(C.TAB_DIR / "hybrid_summary.json", "w"), indent=2)
    ep.to_csv(C.TAB_DIR / "diagnostic_episodes.csv", index=False)
    rc, el, contrib = root_cause_tables(prep, dd, diag)
    print(rc.round(3).to_string())
    print(el.round(3).to_string())
    print(contrib.round(3).to_string())
    print(ep.head(15).to_string())
    plot_root_cause(contrib, rc)
    plot_fleet_daily(prep, dd)
    plot_site_hybrid(prep, dd, preds, rep_site_i)
    np.savez_compressed(C.CACHE_DIR / "decisions.npz", **{k: v for k, v in dd.items() if isinstance(v, np.ndarray)})
    return dd
