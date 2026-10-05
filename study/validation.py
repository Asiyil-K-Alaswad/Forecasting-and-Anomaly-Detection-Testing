"""Detector validation the paper could not do without labels.

1. Synthetic fault injection: realistic faults are written into the *raw* test-period
   telemetry, derived features are recomputed, and both detectors are re-run with
   their frozen weights and thresholds. Recall per fault type is compared with the
   flag rate on the very same site-days before injection (chance level).
2. Operational events as weak labels: LLVD/BLVD operation, mains outages, DG abnormal
   runtime and hardware power-on events are logged by the NMS but are *not*
   detector inputs, so their enrichment among flagged days measures agreement.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from . import anomaly as A
from . import data as D
from . import features as F
from . import forecasting as FC
from . import hybrid as HY
from .plots import save, SERIES, INK_2
from .prepare import Prepared, build

LOAD_COLS = ["dc_power_kw", "dc_power_max_kw", "dc_power_min_kw", "dc_current_a", "dc_current_max_a",
             "dc_current_min_a", "dc_energy_kwh", "energy_in_kwh"]


def _surge(r, g):
    r[LOAD_COLS] *= 1 + g.uniform(0.15, 0.35)


def _drop(r, g):
    f = 1 - g.uniform(0.2, 0.4)
    r[LOAD_COLS] *= f
    r["ran_energy_kwh"] *= f
    r["traffic_gb"] *= 1 - g.uniform(0.3, 0.6)


def _rectifier(r, g):
    r["energy_in_kwh"] *= 1 + g.uniform(0.08, 0.20)


def _overheat(r, g):
    dt = g.uniform(6, 12)
    r["batt_temp_c"] += dt
    r["batt_temp_max_c"] += dt


def _supply(r, g):
    dh, dv = g.uniform(3, 8), g.uniform(1.5, 3.0)
    v = r["dc_power_kw"] * 1000 / r["dc_current_a"]
    r["dc_current_a"] = r["dc_power_kw"] * 1000 / (v - dv)
    r["batt_discharge_h"] = np.minimum(24, r["batt_discharge_h"].fillna(0) + dh)
    r["energy_in_kwh"] *= 1 - dh / 24
    r["mains_outage_h"] = r["mains_outage_h"].fillna(0) + dh


def _meter(r, g):
    r["ran_energy_kwh"] *= 1 - g.uniform(0.2, 0.4)


LABELS = {"A_f": "Forecast residual", "A_a": "Autoencoder", "A_or": "Hybrid OR", "A_and": "Hybrid AND",
          "A_f_adj": "Forecast (fleet-adjusted)", "A_or_adj": "Hybrid OR (fleet-adjusted)",
          "A_and_adj": "Hybrid AND (fleet-adjusted)"}

# name -> (perturbation, possible durations in days, expected primary detector)
FAULTS = {
    "Load surge (non-RAN load, 1 d)": (_surge, [1], "forecast"),
    "Load drop (sector outage, 1-3 d)": (_drop, [1, 2, 3], "forecast"),
    "Rectifier efficiency loss (3-5 d)": (_rectifier, [3, 4, 5], "autoencoder"),
    "Battery overheating (2-4 d)": (_overheat, [2, 3, 4], "autoencoder"),
    "Supply interruption (1-2 d)": (_supply, [1, 2], "autoencoder"),
    "RAN meter drift (3-5 d)": (_meter, [3, 4, 5], "autoencoder"),
}


def choose_events(prep: Prepared, base_dd, rng, per_site=2, gap=45):
    S, T = prep.present.shape
    te = prep.idx["test"]
    ok = prep.observed & ~prep.fc_irr & ~prep.ae_irr & base_dd["scored_f"] & base_dd["scored_a"]
    names = list(FAULTS)
    events = []
    for s in rng.permutation(S):
        cand = np.where(ok[s, te:T - 5])[0] + te
        taken = []
        for d in rng.permutation(cand):
            if len(taken) == per_site:
                break
            if all(abs(d - x) >= gap for x in taken):
                taken.append(int(d))
        for d in taken:
            events.append({"s": int(s), "d": d})
    order = rng.permutation(len(events))
    for k, i in enumerate(order):
        name = names[k % len(names)]
        L = int(rng.choice(FAULTS[name][1]))
        # sustained faults need every affected day observed
        while L > 1 and not prep.observed[events[i]["s"], events[i]["d"]:events[i]["d"] + L].all():
            L -= 1
        events[i].update(fault=name, L=L)
    return pd.DataFrame(events)


def inject(prep: Prepared, events, rng):
    keep = [c for c in prep.panel.columns if not c.startswith(("z_", "irr_", "out_"))]
    p = prep.panel[keep].copy()
    T = len(prep.dates)
    rows = []
    for e in events.itertuples():
        idx = np.arange(e.s * T + e.d, e.s * T + e.d + e.L)
        r = p.loc[idx].copy()
        FAULTS[e.fault][0](r, rng)
        r = D.derive(r)
        p.loc[idx, r.columns] = r
        rows.extend(idx)
    return build(p, prep.static, prep.split, stats=prep.stats)


def run_detectors(prep: Prepared, tau_a, tau_f, tau_f_adj):
    fmodel, sets, _, sc = FC.load_model(prep)
    preds, ref = FC.predict_grid(fmodel, prep, sets, sc)
    amodel = A.load_model()
    pairs = A.window_sets(prep)["score"]
    pairs = pairs[pairs[:, 1] >= prep.idx["val"]]
    err, ferr, _ = A.score(amodel, prep, pairs)
    return HY.decisions(prep, (preds, ref, sc["dc_power_kw"]), err, tau_a, tau_f=tau_f, tau_f_adj=tau_f_adj)


def injection_study(prep: Prepared, base_dd):
    rng = np.random.default_rng(C.SEED)
    events = choose_events(prep, base_dd, rng)
    inj = inject(prep, events, rng)
    dd = run_detectors(inj, base_dd["tau_a"], base_dd["tau_f"], base_dd["tau_f_adj"])
    rows = []
    for e in events.itertuples():
        sl = slice(e.d, e.d + e.L)
        r = {"fault": e.fault, "site": prep.sites[e.s], "day": prep.dates[e.d].date(), "L": e.L}
        for k in LABELS:
            r[k] = bool(dd[k][e.s, sl].any())
            r[k + "_before"] = bool(base_dd[k][e.s, sl].any())
        rows.append(r)
    ev = pd.DataFrame(rows)
    ev.to_csv(C.TAB_DIR / "injection_events.csv", index=False)
    labels = LABELS
    rec = ev.groupby("fault")[[*labels, *[k + "_before" for k in labels]]].mean()
    rec["events"] = ev.groupby("fault").size()
    tot = ev[[*labels, *[k + "_before" for k in labels]]].mean()
    tot["events"] = len(ev)
    rec.loc["All faults"] = tot
    rec = rec.rename(columns={**labels, **{k + "_before": v + " (pre-injection rate)" for k, v in labels.items()}})

    # Day-level false-alarm rate on test days well away from any injected fault
    S, T = prep.present.shape
    near = np.zeros((S, T), bool)
    for e in events.itertuples():
        near[e.s, max(0, e.d - 1):min(T, e.d + e.L + C.FC_WINDOW)] = True
    dom = (dd["scored_f"] | dd["scored_a"]) & ~near
    fa = {labels[k]: float(dd[k][dom].mean()) for k in labels}
    fa_before = {labels[k]: float(base_dd[k][dom].mean()) for k in labels}
    far = pd.DataFrame({"flag rate on unaffected days (injected run)": fa,
                        "flag rate on same days (original run)": fa_before})
    rec.index.name, far.index.name = "fault", "detector"
    rec.to_csv(C.TAB_DIR / "injection_recall.csv")
    far.to_csv(C.TAB_DIR / "injection_false_alarm.csv")
    ts = threshold_sensitivity(prep, base_dd, dd, events, near)
    return rec, far, ts


def threshold_sensitivity(prep: Prepared, base, inj, events, near):
    """Recall on injected faults vs flag rate on unaffected days, for stricter KDE percentiles."""
    T = len(prep.dates)
    tt = np.arange(T)[None, :]
    val = (tt >= prep.idx["val"]) & (tt < prep.idx["test"])
    pools = {"f": base["rf"][val & ~prep.fc_irr & np.isfinite(base["rf"])],
             "a": base["err"][val & ~prep.ae_irr & np.isfinite(base["err"])],
             "f_adj": base["rf_adj"][val & ~prep.fc_irr & np.isfinite(base["rf_adj"])]}
    rows = []
    for q in (95, 99, 99.5):
        tau = {k: F.kde_percentile(v, q) for k, v in pools.items()}

        def flags(dd):
            f = dd["scored_f"] & (np.nan_to_num(dd["rf"], nan=-1) > tau["f"])
            a = dd["scored_a"] & (np.nan_to_num(dd["err"], nan=-1) > tau["a"])
            fa = dd["scored_f"] & (np.nan_to_num(dd["rf_adj"], nan=-1) > tau["f_adj"])
            return {"Forecast residual": f, "Autoencoder": a, "Hybrid OR": f | a, "Hybrid AND": f & a,
                    "Hybrid OR (fleet-adjusted)": fa | a, "Hybrid AND (fleet-adjusted)": fa & a}
        fi, fb = flags(inj), flags(base)
        dom = (base["test"] & (base["scored_f"] | base["scored_a"])) & ~near
        for name in fi:
            hit = np.mean([fi[name][e.s, e.d:e.d + e.L].any() for e in events.itertuples()])
            rows.append({"percentile": q, "detector": name, "recall_injected": float(hit),
                         "flag_rate_unaffected_days": float(fb[name][dom].mean()),
                         "flags_per_1000_site_days": float(fb[name][dom].mean() * 1000)})
    t = pd.DataFrame(rows)
    t.to_csv(C.TAB_DIR / "threshold_sensitivity.csv", index=False)
    return t


def weak_label_study(prep: Prepared, dd):
    S, T = prep.present.shape
    g = lambda c: prep.panel[c].to_numpy().reshape(S, T)  # noqa: E731
    hw = g("hw_poweron_n")
    near = np.zeros_like(hw)
    for k in range(-3, 4):
        if k >= 0:
            near[:, k:] += hw[:, :T - k]
        else:
            near[:, :k] += hw[:, -k:]
    cfg = prep.static["power_config"].to_numpy()[:, None]
    dom = dd["test"] & (dd["scored_f"] | dd["scored_a"])
    events = {
        "Load disconnection (LLVD/BLVD > 0)": ((np.nan_to_num(g("llvd_h")) > 0) | (np.nan_to_num(g("blvd_h")) > 0), dom),
        "Mains outage >= 1 h (grid sites)": (np.nan_to_num(g("mains_outage_h")) >= 1, dom & (cfg == "Grid")),
        "DG abnormal runtime > 0 (DG sites)": (np.nan_to_num(g("dg_abnormal_h")) > 0, dom & (cfg != "Grid")),
        "Hardware power-on within ±3 d": (near > 0, dom),
    }
    labels = LABELS
    rows = []
    for ev_name, (ev, d) in events.items():
        base = ev[d].mean()
        for k, lab in labels.items():
            f = dd[k] & d
            rows.append({"event": ev_name, "detector": lab, "event_days": int((ev & d).sum()),
                         "base_rate": float(base),
                         "P(event | flagged)": float(ev[f].mean()) if f.any() else np.nan,
                         "P(event | not flagged)": float(ev[d & ~dd[k]].mean()),
                         "share_of_event_days_flagged": float(dd[k][ev & d].mean()) if (ev & d).any() else np.nan})
    t = pd.DataFrame(rows)
    t["lift"] = t["P(event | flagged)"] / t["P(event | not flagged)"]
    t.to_csv(C.TAB_DIR / "weak_label_validation.csv", index=False)
    return t


def plot_injection(rec):
    r = rec.drop(index="All faults")
    dets = ["Forecast residual", "Autoencoder", "Hybrid OR", "Hybrid AND"]
    fig, ax = plt.subplots(figsize=(12, 5.2))
    y = np.arange(len(r))
    h = 0.19
    for k, dname in enumerate(dets):
        ax.barh(y + (k - 1.5) * h, r[dname] * 100, height=h * 0.85, color=SERIES[k], label=dname)
    for yy, v in zip(y, r["Hybrid OR (pre-injection rate)"] * 100):
        ax.plot([v, v], [yy - 2 * h, yy + 2 * h], color=INK_2, lw=1.2)
    ax.plot([], [], color=INK_2, lw=1.2, label="OR flag rate on the same days before injection (chance)")
    ax.set_yticks(y, r.index)
    ax.set_xlabel("Injected faults detected (%)")
    ax.set_xlim(0, 105)
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=5, fontsize=8)
    ax.set_title("Synthetic fault injection: detection rate by fault type and detector", loc="left")
    fig.tight_layout()
    return save(fig, "fig11_injection_recall.png")


def run(prep: Prepared):
    dz = np.load(C.CACHE_DIR / "decisions.npz")
    base = {k: dz[k] for k in dz.files}
    hs = json.load(open(C.TAB_DIR / "hybrid_summary.json"))
    base["tau_a"], base["tau_f"] = hs["tau_a"], hs["tau_f_abs_residual_z"]
    base["tau_f_adj"] = hs["tau_f_adj_abs_residual_z"]
    rec, far, ts = injection_study(prep, base)
    print(rec.round(3).to_string())
    print(far.round(4).to_string())
    print(ts.round(4).to_string())
    plot_injection(rec)
    wl = weak_label_study(prep, base)
    print(wl.round(4).to_string())
