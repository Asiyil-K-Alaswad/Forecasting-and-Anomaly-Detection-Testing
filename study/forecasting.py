"""LSTM-based DC load forecasting (paper Sec. III-F, IV-A).

Window-level ("instance") normalisation
---------------------------------------
Site loads in this fleet are non-stationary: seasonal cooling load, steady growth,
and step changes when radio hardware is added (one site runs at 4.7x its
training-period mean in the test period). A network fed site-level z-scores
saturates on such inputs and reverts towards the training-year level. Each window
is therefore scaled by its own recent level (mean of the last 7 input days), as in
RevIN / DeepAR: load, traffic and RAN energy enter as relative deviations from that
level, the LSTM predicts relative deviations, and the output is rescaled to kW.
Weather and calendar inputs keep their global scaling. A first run without this
step is kept as an ablation (outputs/tables/ablation/).
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

from . import config as C
from . import models as M
from .plots import save, SERIES, INK_2
from .prepare import FC_INPUTS, Prepared, shift_left, window_all

W, H = C.FC_WINDOW, C.FC_HORIZON
REF_DAYS = 7
CLIP = 10.0
# positions in FC_INPUTS that are rescaled by the window's own level
LOCAL = {0: "dc_power_kw", 3: "traffic_gb", 4: "ran_energy_kwh"}
SCALE_FILE = "forecast_window_scaling.json"


def window_sets(prep: Prepared, w=W):
    S, T, _ = prep.fc.shape
    in_ok = window_all(np.isfinite(prep.fc).all(-1), w)
    p_obs = prep.present & np.isfinite(prep.fc[..., 0])
    tgt_all = shift_left(window_all(p_obs, H), H)
    clean = shift_left(window_all(~prep.fc_irr, w + H), H)
    t = np.arange(T)[None, :]
    v, te = prep.idx["val"], prep.idx["test"]
    sets = {
        "train": in_ok & tgt_all & clean & (t + H < v),
        "val": in_ok & tgt_all & clean & (t + 1 >= v) & (t + H < te),
        "score": in_ok & (t + 1 >= v) & (t + 1 <= T - 1),
        "last": in_ok & (t == T - 1),
    }
    sets["n_train_all"] = int((in_ok & tgt_all & (t + H < v)).sum())
    return {k: (np.argwhere(m) if isinstance(m, np.ndarray) else m) for k, m in sets.items()}


def _ratios(prep, x):
    """site-relative z -> ratio to site training mean, for the locally scaled inputs."""
    sig = prep.stats["sigma"]
    return {j: 1.0 + sig[name] * x[..., j] for j, name in LOCAL.items()}


def encode(prep: Prepared, pairs, sc, w=W, with_y=True):
    """Gather windows for (site, origin) pairs -> (x, y, ref) with window-level scaling.

    ref = mean load over the last REF_DAYS input days as a ratio to the site mean;
    y   = (P(t+h) / (site_mean * ref) - 1) / sc[target]
    """
    fc = prep.fc
    T = fc.shape[1]
    s, t = pairs[:, 0], pairs[:, 1]
    x = fc[s[:, None], t[:, None] + np.arange(-w + 1, 1)].copy()
    r = _ratios(prep, x)
    ref = {}
    for j, name in LOCAL.items():
        lvl = np.nanmean(r[j][:, -REF_DAYS:], axis=1)
        lvl = np.where(np.abs(lvl) > 1e-3, lvl, np.nan)
        x[..., j] = (r[j] / lvl[:, None] - 1.0) / sc[name]
        ref[j] = lvl
    x = np.clip(np.nan_to_num(x, nan=0.0), -CLIP, CLIP).astype(np.float32)
    y = None
    if with_y:
        tt = np.minimum(t[:, None] + np.arange(1, H + 1), T - 1)
        ry = 1.0 + prep.stats["sigma"]["dc_power_kw"] * fc[s[:, None], tt, 0]
        y = ((ry / ref[0][:, None] - 1.0) / sc["dc_power_kw"]).astype(np.float32)
    return x, y, ref[0]


def fit_scaling(prep: Prepared, sets, n=100_000, w=W, fname=SCALE_FILE):
    """Fleet scale of window-relative deviations (training windows only)."""
    rng = np.random.default_rng(C.SEED)
    tr = sets["train"][rng.choice(len(sets["train"]), min(n, len(sets["train"])), replace=False)]
    ones = {name: 1.0 for name in LOCAL.values()}
    x, y, _ = encode(prep, tr, ones, w)
    sc = {}
    for j, name in LOCAL.items():
        v = x[..., j].ravel()
        lo, hi = np.percentile(v, [1, 99])
        sc[name] = float(np.std(np.clip(v, lo, hi)))
    lo, hi = np.percentile(y[:, 0], [1, 99])
    sc["target_h1"] = float(np.std(np.clip(y[:, 0], lo, hi)))
    json.dump(sc, open(C.TAB_DIR / fname, "w"), indent=2)
    return sc


def make_batch_fn(prep, pairs, sc, w=W, with_y=True):
    def fn(idx):
        x, y, _ = encode(prep, pairs[idx], sc, w, with_y)
        return [torch.from_numpy(x)] + ([torch.from_numpy(y)] if with_y else [])
    return fn


def fit(prep: Prepared, w=W, epochs=C.FC_EPOCHS, tag="lstm_forecaster"):
    M.seed_everything()
    sets = window_sets(prep, w)
    sc = fit_scaling(prep, sets, w=w, fname=SCALE_FILE if tag == "lstm_forecaster" else f"{tag}_scaling.json")
    tr, va = sets["train"], sets["val"]
    print(f"[forecast] windows: train {len(tr):,} (clean partition; {sets['n_train_all']:,} before "
          f"screening), val {len(va):,}; window scales {json.dumps({k: round(v, 4) for k, v in sc.items()})}")
    model = M.LSTMForecaster(len(FC_INPUTS))
    xv, yv = make_batch_fn(prep, va, sc, w)(np.arange(len(va)))

    def loss_fn(m, x, y):
        return torch.mean((m(x) - y) ** 2)

    def val_fn(m):
        return float(torch.mean((m(xv) - yv) ** 2))

    hist = M.train(model, make_batch_fn(prep, tr, sc, w), len(tr), val_fn, epochs, C.FC_LR, C.FC_BATCH,
                   loss_fn, tag)
    C.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), C.MODEL_DIR / f"{tag}.pt")
    pd.DataFrame(hist).to_csv(C.TAB_DIR / f"{tag}_history.csv", index=False)
    return model, sets, hist, sc


def load_model(prep, tag="lstm_forecaster"):
    model = M.LSTMForecaster(len(FC_INPUTS))
    model.load_state_dict(torch.load(C.MODEL_DIR / f"{tag}.pt"))
    hist = pd.read_csv(C.TAB_DIR / f"{tag}_history.csv").to_dict("records")
    sc = json.load(open(C.TAB_DIR / SCALE_FILE))
    return model, window_sets(prep), hist, sc


def _grid(prep, pairs, out, ref, sc):
    """Place window outputs into kW grids: preds[s, t, h-1] = forecast of day t+h issued at t."""
    S, T, _ = prep.fc.shape
    preds = np.full((S, T, H), np.nan, np.float32)
    ref_kw = np.full((S, T), np.nan, np.float32)
    lvl_kw = ref * prep.site_mean_kw[pairs[:, 0]]
    preds[pairs[:, 0], pairs[:, 1]] = lvl_kw[:, None] * (1.0 + sc["dc_power_kw"] * out)
    ref_kw[pairs[:, 0], pairs[:, 1]] = lvl_kw
    return preds, ref_kw


def predict_grid(model, prep: Prepared, sets, sc, w=W):
    pairs = np.unique(np.concatenate([sets["score"], sets["last"]]), axis=0)
    out, refs = [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(pairs), 4096):
            x, _, ref = encode(prep, pairs[i:i + 4096], sc, w, with_y=False)
            out.append(model(torch.from_numpy(x)).numpy())
            refs.append(ref)
    return _grid(prep, pairs, np.concatenate(out), np.concatenate(refs), sc)


def ridge_baseline(prep, sets, sc, w=W, n_max=150_000):
    rng = np.random.default_rng(C.SEED)
    tr = sets["train"]
    tr = tr[rng.choice(len(tr), min(n_max, len(tr)), replace=False)]
    x, y, _ = encode(prep, tr, sc, w)
    reg = Ridge(alpha=1.0).fit(x.reshape(len(tr), -1), y)
    pairs = sets["score"]
    xs, _, ref = encode(prep, pairs, sc, w, with_y=False)
    preds, _ = _grid(prep, pairs, reg.predict(xs.reshape(len(pairs), -1)), ref, sc)
    return preds


def _metrics(y, p):
    e = p - y
    ss_res, ss_tot = np.sum(e ** 2), np.sum((y - y.mean()) ** 2)
    return {"MAE_W": float(np.mean(np.abs(e)) * 1000), "RMSE_W": float(np.sqrt(np.mean(e ** 2)) * 1000),
            "MAPE_%": float(np.mean(np.abs(e) / np.abs(y)) * 100), "R2": float(1 - ss_res / ss_tot)}


def _lead(arr, h):
    """Align a forecast/series issued at t to its target day t+h."""
    out = np.full(arr.shape[:2], np.nan)
    out[:, h:] = arr[:, :-h]
    return out


def evaluate(prep: Prepared, preds, ridge):
    """Test-period metrics on day-ahead (h=1) forecasts + horizon curve + per-group breakdown."""
    S, T = prep.present.shape
    te = prep.idx["test"]
    actual = np.where(prep.observed, prep.to_grid("dc_power_kw"), np.nan)
    f1, r1 = _lead(preds[..., 0], 1), _lead(ridge[..., 0], 1)
    pers, snaive = _lead(actual, 1), _lead(actual, 7)
    ma7 = pd.DataFrame(actual.T).rolling(7, min_periods=5).mean().shift(1).to_numpy().T

    day = np.arange(T)[None, :]
    m_all = (day >= te) & np.isfinite(actual)
    for f in (f1, r1, pers, snaive, ma7):
        m_all &= np.isfinite(f)
    m_reg = m_all & ~prep.fc_irr
    rows = []
    models = [("LSTM (this study)", f1), ("Linear ARX (ridge, same inputs)", r1),
              ("Persistence  P(t)", pers), ("Seasonal naive  P(t-6)", snaive), ("7-day mean", ma7)]
    for name, f in models:
        for subset, mm in [("all test days", m_all), ("regular test days", m_reg)]:
            r = _metrics(actual[mm], f[mm])
            r.update(model=name, subset=subset, n=int(mm.sum()))
            rows.append(r)
    res = pd.DataFrame(rows)
    pmse = res.query("model == 'Persistence  P(t)'").set_index("subset")["RMSE_W"] ** 2
    res["skill_vs_persistence"] = 1 - res["RMSE_W"] ** 2 / res["subset"].map(pmse)

    # Per-site R2: within-site explained variance (pooled R2 is inflated by between-site spread)
    site_r2 = {}
    for name, f in models:
        r2 = []
        for s in range(S):
            mm = m_all[s]
            if mm.sum() >= 30:
                y = actual[s, mm]
                r2.append(1 - np.sum((f[s, mm] - y) ** 2) / np.sum((y - y.mean()) ** 2))
        site_r2[name] = float(np.median(r2))
    res["median_site_R2"] = res["model"].map(site_r2)
    res = res[["model", "subset", "n", "MAE_W", "RMSE_W", "MAPE_%", "R2", "median_site_R2",
               "skill_vs_persistence"]]

    hz = []
    for h in range(1, H + 1):
        fh, rh, ph = _lead(preds[..., h - 1], h), _lead(ridge[..., h - 1], h), _lead(actual, h)
        mm = (day >= te) & np.isfinite(actual) & np.isfinite(fh) & np.isfinite(ph) & np.isfinite(rh)
        a = actual[mm]
        hz.append({"h": h, "LSTM": np.mean(np.abs(fh[mm] - a)) * 1000,
                   "Linear ARX": np.mean(np.abs(rh[mm] - a)) * 1000,
                   "Persistence": np.mean(np.abs(ph[mm] - a)) * 1000,
                   "LSTM MAPE %": np.mean(np.abs(fh[mm] - a) / a) * 100, "n": int(mm.sum())})
    hz = pd.DataFrame(hz)

    grp = []
    for col in ["power_config", "site_type"]:
        for g, sub in prep.static.groupby(col):
            si = np.isin(prep.sites, sub.index)
            mm = m_all & si[:, None]
            if mm.sum() < 500:
                continue
            r, rp = _metrics(actual[mm], f1[mm]), _metrics(actual[mm], pers[mm])
            grp.append({"group": f"{col}={g}", "sites": int(si.sum()), "n": int(mm.sum()),
                        "MAE_W": r["MAE_W"], "RMSE_W": r["RMSE_W"], "MAPE_%": r["MAPE_%"], "R2": r["R2"],
                        "skill_vs_persistence": 1 - (r["RMSE_W"] / rp["RMSE_W"]) ** 2})
    grp = pd.DataFrame(grp)

    res.to_csv(C.TAB_DIR / "table3_forecast_metrics.csv", index=False)
    hz.to_csv(C.TAB_DIR / "forecast_horizon_mae.csv", index=False)
    grp.to_csv(C.TAB_DIR / "forecast_by_group.csv", index=False)
    return res, hz, grp


def plot_site_forecast(prep: Prepared, preds, site_i, name="fig07_forecast_site.png"):
    T = len(prep.dates)
    te = prep.idx["test"]
    actual = np.where(prep.observed[site_i], prep.to_grid("dc_power_kw")[site_i], np.nan)
    f1 = _lead(preds[site_i:site_i + 1, :, 0], 1)[0]
    start = te - 21
    d = prep.dates
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(d[start:], actual[start:], color=SERIES[0], lw=1.8, label="Actual")
    ax.plot(d[te:], f1[te:], color=SERIES[1], lw=1.4, label="Day-ahead forecast (test)")
    fut = preds[site_i, T - 1]
    if np.isfinite(fut).all():
        fd = pd.date_range(d[-1] + pd.Timedelta(days=1), periods=H)
        ax.plot([d[-1], *fd], [actual[-1], *fut], color=SERIES[2], lw=1.8, marker="o", markersize=4,
                label=f"Future {H}-day forecast")
    ax.axvline(d[te], color=INK_2, lw=0.8)
    ax.text(d[te], ax.get_ylim()[1], "  test period →", va="top", fontsize=8, color=INK_2)
    ax.set_ylabel("DC load power (kW)")
    ax.set_title(f"LSTM forecast: actual, day-ahead and future DC load — site {prep.sites[site_i]}",
                 loc="left")
    ax.legend(loc="lower left", ncol=3)
    return save(fig, name)


def plot_horizon(hz):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for i, c in enumerate(["LSTM", "Linear ARX", "Persistence"]):
        ax.plot(hz["h"], hz[c], marker="o", markersize=4, color=SERIES[i], lw=1.8, label=c)
        ax.text(hz["h"].iloc[-1] + 0.12, hz[c].iloc[-1], f"{hz[c].iloc[-1]:.0f}", va="center",
                fontsize=8, color=INK_2)
    ax.set_xlabel("Lead time (days)")
    ax.set_ylabel("MAE (W)")
    ax.set_xlim(0.7, H + 0.7)
    ax.set_title("Forecast error by lead time (test period, all sites)", loc="left")
    ax.legend()
    return save(fig, "fig07b_horizon_mae.png")


def run(prep: Prepared, rep_site_i: int, retrain=True):
    model, sets, hist, sc = fit(prep) if retrain else load_model(prep)
    preds, ref_kw = predict_grid(model, prep, sets, sc)
    ridge = ridge_baseline(prep, sets, sc)
    res, hz, grp = evaluate(prep, preds, ridge)
    print(res.round(4).to_string())
    print(hz.round(2).to_string())
    print(grp.round(4).to_string())
    plot_site_forecast(prep, preds, rep_site_i)
    plot_horizon(hz)
    np.save(C.CACHE_DIR / "fc_preds_kw.npy", preds)
    np.save(C.CACHE_DIR / "fc_ref_kw.npy", ref_kw)
    meta = {"train_windows": int(len(sets["train"])), "train_windows_before_screening": sets["n_train_all"],
            "val_windows": int(len(sets["val"])), "best_epoch": int(pd.DataFrame(hist)["val_loss"].idxmin() + 1),
            "window_scaling": sc}
    json.dump(meta, open(C.TAB_DIR / "forecast_meta.json", "w"), indent=2)
    return preds, res


def window_sensitivity(prep: Prepared, windows=(14, 28, 60), epochs=10):
    """Look-back length ablation (paper: 60 steps). Day-ahead metrics on a common set of test days."""
    T = len(prep.dates)
    te = prep.idx["test"]
    actual = np.where(prep.observed, prep.to_grid("dc_power_kw"), np.nan)
    f1s, hist = {}, {}
    for w in windows:
        model, sets, h, sc = fit(prep, w=w, epochs=epochs, tag=f"sens_w{w}")
        preds, _ = predict_grid(model, prep, sets, sc, w=w)
        f1s[w] = _lead(preds[..., 0], 1)
        hist[w] = int(pd.DataFrame(h)["val_loss"].idxmin() + 1)
    pers = _lead(actual, 1)
    m = (np.arange(T)[None, :] >= te) & np.isfinite(actual) & np.isfinite(pers)
    for f in f1s.values():
        m &= np.isfinite(f)
    rp = _metrics(actual[m], pers[m])
    rows = []
    for w, f in f1s.items():
        r = _metrics(actual[m], f[m])
        rows.append({"window_days": w, "epochs": epochs, "best_epoch": hist[w], "n": int(m.sum()), **r,
                     "skill_vs_persistence": 1 - (r["RMSE_W"] / rp["RMSE_W"]) ** 2})
    t = pd.DataFrame(rows)
    t.to_csv(C.TAB_DIR / "forecast_window_sensitivity.csv", index=False)
    print(t.round(4).to_string())
    return t
