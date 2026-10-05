"""LSTM autoencoder anomaly detection on power-system health indicators (Sec. III-G, IV-B).

Each site-day is scored causally: the reconstruction error of the *last* step of
the 14-day window that ends on that day, so a day is flagged with information
available at the end of that day (real-time capable, as in the paper's goal).
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from . import config as C
from . import features as F
from . import models as M
from .plots import save, SERIES, INK_2, CRITICAL, WARNING
from .prepare import AE_INPUTS, Prepared, window_all

W = C.AE_WINDOW


def window_sets(prep: Prepared):
    S, T = prep.present.shape
    ok = window_all(prep.present, W)
    clean = window_all(~prep.ae_irr, W)
    t = np.arange(T)[None, :]
    v, te = prep.idx["val"], prep.idx["test"]
    return {
        "train": np.argwhere(ok & clean & (t < v)),
        "val": np.argwhere(ok & clean & (t >= v + W - 1) & (t < te)),
        "score": np.argwhere(ok),
    }


def make_batch_fn(prep: Prepared, pairs):
    ae = prep.ae
    offs = np.arange(-W + 1, 1)

    def fn(idx):
        s, t = pairs[idx, 0], pairs[idx, 1]
        x = ae[s[:, None], t[:, None] + offs]
        m = np.isfinite(x).astype(np.float32)
        return [torch.from_numpy(np.nan_to_num(x, nan=0.0)), torch.from_numpy(m)]
    return fn


def fit(prep: Prepared, tag="lstm_autoencoder"):
    M.seed_everything()
    sets = window_sets(prep)
    print(f"[ae] windows: train {len(sets['train']):,} (clean), val {len(sets['val']):,}, "
          f"score {len(sets['score']):,}")
    model = M.LSTMAutoencoder(len(AE_INPUTS))
    xv, mv = make_batch_fn(prep, sets["val"])(np.arange(len(sets["val"])))

    def loss_fn(m, x, mask):
        return M.masked_mse(m(x), x, mask)

    def val_fn(m):
        return float(M.masked_mse(m(xv), xv, mv))

    hist = M.train(model, make_batch_fn(prep, sets["train"]), len(sets["train"]), val_fn,
                   C.AE_EPOCHS, C.AE_LR, C.AE_BATCH, loss_fn, tag)
    torch.save(model.state_dict(), C.MODEL_DIR / f"{tag}.pt")
    pd.DataFrame(hist).to_csv(C.TAB_DIR / f"{tag}_history.csv", index=False)
    return model, sets


@torch.no_grad()
def score(model, prep: Prepared, pairs=None, batch=4096, collect_test_metrics=False):
    """Return day error [S,T], per-feature last-step error [S,T,F] and optional Table IV metrics."""
    model.eval()
    S, T, nf = prep.ae.shape
    if pairs is None:
        pairs = window_sets(prep)["score"]
    fn = make_batch_fn(prep, pairs)
    err = np.full((S, T), np.nan, np.float32)
    ferr = np.full((S, T, nf), np.nan, np.float32)
    te = prep.idx["test"]
    acc = {"abs": 0.0, "sq": 0.0, "n": 0, "y": 0.0, "y2": 0.0}
    for i in range(0, len(pairs), batch):
        idx = np.arange(i, min(i + batch, len(pairs)))
        x, m = fn(idx)
        r = model(x)
        se = ((r - x) ** 2 * m).numpy()
        last_se, last_m = se[:, -1], m[:, -1].numpy()
        s, t = pairs[idx, 0], pairs[idx, 1]
        err[s, t] = last_se.sum(1) / np.maximum(last_m.sum(1), 1)
        ferr[s, t] = np.where(last_m > 0, last_se, np.nan)
        if collect_test_metrics:
            sel = t >= te + W - 1  # windows fully inside the test period
            if sel.any():
                mm = m.numpy()[sel] > 0
                xx, rr = x.numpy()[sel][mm], r.numpy()[sel][mm]
                acc["abs"] += np.abs(rr - xx).sum()
                acc["sq"] += ((rr - xx) ** 2).sum()
                acc["n"] += mm.sum()
                acc["y"] += xx.sum()
                acc["y2"] += (xx ** 2).sum()
    metrics = None
    if collect_test_metrics and acc["n"]:
        n = acc["n"]
        var = acc["y2"] / n - (acc["y"] / n) ** 2
        mse = acc["sq"] / n
        metrics = {"MAE": acc["abs"] / n, "MSE": mse, "RMSE": np.sqrt(mse), "R2": 1 - mse / var}
    return err, ferr, metrics


def threshold(prep: Prepared, err):
    """95th percentile of the KDE fitted to errors on clean, held-out validation days."""
    v, te = prep.idx["val"], prep.idx["test"]
    T = err.shape[1]
    tt = np.arange(T)[None, :]
    ok = ~prep.ae_irr & np.isfinite(err) & prep.observed
    sel = (tt >= v) & (tt < te) & ok
    tau = F.kde_percentile(err[sel], C.THRESHOLD_PERCENTILE)
    sel_tr = (tt < v) & ok
    tau_train = F.kde_percentile(err[sel_tr], C.THRESHOLD_PERCENTILE)
    return tau, tau_train, err[sel]


def plot_site_error(prep: Prepared, err, tau, i, name="fig08_ae_error_site.png"):
    d = prep.dates
    e = np.where(prep.observed[i], err[i], np.nan)
    flag = e > tau
    te = prep.idx["test"]
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.plot(d, e, color=SERIES[0], lw=1.2, label="Reconstruction error")
    ax.scatter(d[flag], e[flag], marker="x", s=28, color=CRITICAL, lw=1.4, zorder=3,
               label=f"Anomalies ({int(flag.sum())})")
    ax.axhline(tau, color=WARNING, lw=1.4, label=f"Threshold (KDE {C.THRESHOLD_PERCENTILE}th pct = {tau:.3f})")
    ax.axvline(d[te], color=INK_2, lw=0.8)
    ax.text(d[te], np.nanmax(e), "  test period →", va="top", fontsize=8, color=INK_2)
    ax.set_yscale("log")
    ax.set_ylabel("Error (masked MSE, log)")
    ax.set_title(f"LSTM autoencoder reconstruction error — site {prep.sites[i]}", loc="left")
    ax.legend(loc="upper left", ncol=3)
    return save(fig, name)


def plot_error_kde(val_err, tau):
    grid, pdf, cdf, _ = F.binned_kde(np.log10(val_err[val_err > 0]))
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.fill_between(10 ** grid, pdf, color=SERIES[0], alpha=0.1, lw=0)
    ax.plot(10 ** grid, pdf, color=SERIES[0], lw=1.8)
    ax.axvline(tau, color=WARNING, lw=1.4)
    ax.text(tau, pdf.max() * 0.95, f"  τa = {tau:.3f}\n  (95th pct)", fontsize=8, color=INK_2, va="top")
    ax.set_xscale("log")
    ax.set_xlabel("Reconstruction error (log scale)")
    ax.set_ylabel("KDE density (of log10 error)")
    ax.set_title("Validation-day reconstruction error distribution", loc="left")
    return save(fig, "fig08b_ae_error_kde.png")


def load_model(tag="lstm_autoencoder"):
    model = M.LSTMAutoencoder(len(AE_INPUTS))
    model.load_state_dict(torch.load(C.MODEL_DIR / f"{tag}.pt"))
    return model


def run(prep: Prepared, rep_site_i: int, retrain=True):
    if retrain:
        model, sets = fit(prep)
    else:
        model, sets = load_model(), window_sets(prep)
    err, ferr, metrics = score(model, prep, sets["score"], collect_test_metrics=True)
    tau, tau_train, val_err = threshold(prep, err)
    te = prep.idx["test"]
    test = np.zeros_like(err, bool)
    test[:, te:] = True
    scored = test & np.isfinite(err) & prep.observed
    n_flag = int(((err > tau) & scored).sum())
    out = {**{k: float(v) for k, v in metrics.items()}, "threshold": float(tau),
           "threshold_if_fitted_on_training_days": float(tau_train),
           "test_site_days_scored": int(scored.sum()), "test_anomalies": n_flag,
           "test_anomaly_rate": n_flag / int(scored.sum()),
           "train_windows": int(len(sets["train"])), "val_windows": int(len(sets["val"]))}
    print(json.dumps(out, indent=2))
    json.dump(out, open(C.TAB_DIR / "table4_autoencoder_metrics.json", "w"), indent=2)
    np.save(C.CACHE_DIR / "ae_err.npy", err)
    np.save(C.CACHE_DIR / "ae_ferr.npy", ferr)
    plot_site_error(prep, err, tau, rep_site_i)
    plot_error_kde(val_err, tau)
    return err, ferr, tau
