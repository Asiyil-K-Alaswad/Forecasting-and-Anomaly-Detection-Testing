"""Run the full study end-to-end, or individual stages.

    python run_study.py                 # everything
    python run_study.py eda forecast    # selected stages (later stages reuse cached results)
"""
import json
import sys

import numpy as np

from study import config as C
from study.prepare import prepare

STAGES = ["prepare", "eda", "forecast", "autoencoder", "hybrid", "validate", "report"]
# "forecast-eval" / "autoencoder-eval" re-evaluate saved weights without retraining;
# "sensitivity" runs the look-back window ablation (14 / 28 / 60 days).


def representative_site(prep):
    """Fixed rule, applied before any modelling (the paper studies one panel).

    Outdoor grid site, battery-temperature sensor present, the most observed days,
    and mean load closest to the fleet median among those.
    """
    st = prep.static
    obs = prep.present.sum(1)
    has_bt = np.isfinite(prep.ae[..., 4]).mean(1) > 0.9
    cand = (st["site_type"].to_numpy() == "Outdoor") & (st["power_config"].to_numpy() == "Grid") & has_bt
    cand &= obs >= np.max(obs[cand])
    med = np.median(prep.site_mean_kw)
    i = int(np.argmin(np.where(cand, np.abs(prep.site_mean_kw - med), np.inf)))
    return i


def main(argv):
    stages = argv or STAGES
    for d in (C.FIG_DIR, C.TAB_DIR, C.MODEL_DIR, C.CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)
    prep = prepare(force="prepare" in stages)
    rep = representative_site(prep)
    print(f"[study] {len(prep.sites):,} sites x {len(prep.dates)} days; representative site "
          f"{prep.sites[rep]}; split {json.dumps({k: str(v.date()) for k, v in prep.split.items()})}")

    if "eda" in stages:
        from study import stage_eda
        stage_eda.run(prep, rep)
    if "forecast" in stages or "forecast-eval" in stages:
        from study import forecasting
        forecasting.run(prep, rep, retrain="forecast" in stages)
    if "sensitivity" in stages:
        from study import forecasting
        forecasting.window_sensitivity(prep)
    if "autoencoder" in stages or "autoencoder-eval" in stages:
        from study import anomaly
        anomaly.run(prep, rep, retrain="autoencoder" in stages)
    if "hybrid" in stages:
        from study import hybrid
        hybrid.run(prep, rep)
    if "validate" in stages:
        from study import validation
        validation.run(prep)
    if "report" in stages:
        from study import summary
        summary.run(prep, rep)


if __name__ == "__main__":
    main(sys.argv[1:])
