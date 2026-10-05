"""Turn the cleaned panel into model-ready tensors, masks and the data partition."""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from . import features as F

FC_INPUTS = ["z_" + c for c in C.FORECAST_FEATURES] + C.CALENDAR_FEATURES
AE_INPUTS = ["z_" + c for c in C.AE_FEATURES]


@dataclass
class Prepared:
    panel: pd.DataFrame            # selected sites, complete calendar, sorted (site, date)
    static: pd.DataFrame
    split: dict
    stats: dict
    sites: np.ndarray
    dates: pd.DatetimeIndex
    fc: np.ndarray                 # [S, T, len(FC_INPUTS)]  (NaN = missing)
    ae: np.ndarray                 # [S, T, len(AE_INPUTS)]
    present: np.ndarray            # [S, T] rectifier export present (load observed)
    fc_irr: np.ndarray             # [S, T] consensus outlier in forecasting telemetry
    ae_irr: np.ndarray             # [S, T] consensus outlier in health indicators
    site_mean_kw: np.ndarray       # [S]
    idx: dict = field(default_factory=dict)  # val / test start indices
    screen_info: pd.DataFrame = None
    prep_log: pd.DataFrame = None
    observed: np.ndarray = None    # [S, T] load genuinely reported (not interpolated)

    def to_grid(self, col):
        return self.panel[col].to_numpy().reshape(len(self.sites), len(self.dates))


def _tensor(df, cols, S, T):
    return df[cols].to_numpy(np.float32).reshape(S, T, len(cols))


def build(panel, static, split, stats=None, prep_log=None):
    """Normalise / screen / tensorise a panel. `stats` is re-used for injected copies."""
    panel = panel.sort_values(["site", "date"]).reset_index(drop=True)
    sites = panel["site"].unique()
    dates = pd.DatetimeIndex(np.sort(panel["date"].unique()))
    S, T = len(sites), len(dates)
    assert len(panel) == S * T, "panel must be a complete site x day grid"
    if stats is None:
        stats = F.fit_normaliser(panel, split)
    z = F.transform(panel, stats)
    flags, info = F.screen_outliers(panel, z, split)
    panel = pd.concat([panel, z], axis=1)
    panel["irr_fc"] = flags[["dc_power_kw", "traffic_gb", "ran_energy_kwh"]].any(axis=1)
    panel["irr_ae"] = flags[C.AE_FEATURES].any(axis=1)
    for c in F.SCREENED:
        panel["out_" + c] = flags[c]
    site_mean = stats["site_center"]["dc_power_kw"].reindex(sites).to_numpy()
    return Prepared(
        panel=panel, static=static.loc[sites], split=split, stats=stats, sites=sites, dates=dates,
        fc=_tensor(panel, FC_INPUTS, S, T), ae=_tensor(panel, AE_INPUTS, S, T),
        present=panel["dc_power_kw"].notna().to_numpy().reshape(S, T),
        fc_irr=panel["irr_fc"].to_numpy().reshape(S, T),
        ae_irr=panel["irr_ae"].to_numpy().reshape(S, T),
        site_mean_kw=site_mean,
        idx={"val": int(dates.get_loc(split["val_start"])), "test": int(dates.get_loc(split["test_start"]))},
        screen_info=info, prep_log=prep_log,
        observed=(panel["observed"] if "observed" in panel else panel["dc_power_kw"].notna())
        .to_numpy().reshape(S, T),
    )


def prepare(force=False) -> Prepared:
    path = C.CACHE_DIR / "prepared.pkl"
    if path.exists() and not force:
        with open(path, "rb") as f:
            return pickle.load(f)
    panel, static, log = D.build_panel()
    split = F.split_dates(panel)
    sites = F.select_sites(panel, split)
    panel = panel[panel["site"].isin(sites)].reset_index(drop=True)
    panel["observed"] = panel["dc_power_kw"].notna()
    panel = F.interpolate_short_gaps(panel, C.SELECTED_FEATURES)
    log = pd.concat([log, pd.DataFrame([
        {"step": "site selection", "detail": f"{len(sites):,} of {static.shape[0]:,} sites have "
                                             f">= {int(C.MIN_VALID_DAYS * C.TRAIN_FRACTION)} observed training days "
                                             f"and >= 30 test days", "rows_after": len(panel)},
        {"step": "missing-value handling", "detail": f"gaps <= {C.MAX_INTERP_GAP} days linearly interpolated "
                                                     "inside each site; longer gaps break sequences",
         "rows_after": len(panel)}])], ignore_index=True)
    prep = build(panel, static, split, prep_log=log)
    with open(path, "wb") as f:
        pickle.dump(prep, f)
    return prep


# --------------------------------------------------------------------------
# Window indexing helpers
# --------------------------------------------------------------------------
def window_all(mask, w):
    """ok[s, t] = mask[s, t-w+1 .. t] all True (False where t < w-1)."""
    S, T = mask.shape
    cs = np.concatenate([np.zeros((S, 1), np.int32), np.cumsum(~mask, axis=1, dtype=np.int32)], axis=1)
    ok = np.zeros((S, T), bool)
    ok[:, w - 1:] = (cs[:, w:] - cs[:, :T - w + 1]) == 0
    return ok


def shift_left(a, k, fill=False):
    """b[:, t] = a[:, t + k]."""
    b = np.full_like(a, fill)
    if k == 0:
        return a.copy()
    b[:, :-k] = a[:, k:]
    return b
