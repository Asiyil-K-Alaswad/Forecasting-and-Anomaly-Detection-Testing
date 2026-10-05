"""Stage: exploratory analysis, feature selection and partitioning (Sec. III-B..E)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from . import eda
from . import features as F
from .plots import save, SERIES, INK_2
from .prepare import Prepared, window_all


def plot_site_signals(prep: Prepared, i):
    """Fig. 3 analogue: raw exported signals for the representative site."""
    p = prep.panel.iloc[i * len(prep.dates):(i + 1) * len(prep.dates)]
    rows = [("DC load power (kW)", ["dc_power_kw", "dc_power_max_kw", "dc_power_min_kw"], ["Avg", "Max", "Min"]),
            ("Temperature (°C)", ["amb_temp_c", "batt_temp_c"], ["Ambient (weather)", "Battery"]),
            ("Energy (kWh/day)", ["dc_energy_kwh", "ran_energy_kwh"], ["DC load energy", "RAN board energy (NMS)"]),
            ("Traffic (GB/day)", ["traffic_gb"], ["Traffic"])]
    fig, axes = plt.subplots(len(rows), 1, figsize=(12, 8.5), sharex=True)
    for ax, (yl, cols, labs) in zip(axes, rows):
        for k, (c, lab) in enumerate(zip(cols, labs)):
            ax.plot(p["date"], p[c], color=SERIES[k], lw=1.3, label=lab)
        ax.set_ylabel(yl)
        if len(cols) > 1:
            ax.legend(loc="upper left", ncol=len(cols), fontsize=8)
    axes[0].set_title(f"Daily exported signals — representative site {prep.sites[i]} "
                      f"({prep.static.iloc[i]['site_type']}, {prep.static.iloc[i]['power_config']})", loc="left")
    fig.tight_layout()
    return save(fig, "fig03_site_signals.png")


def partition_table(prep: Prepared):
    from .forecasting import window_sets
    fc = window_sets(prep)
    S, T = prep.present.shape
    t = np.arange(T)[None, :]
    ae_all = window_all(prep.present, C.AE_WINDOW) & (t < prep.idx["val"])
    ae_clean = ae_all & window_all(~prep.ae_irr, C.AE_WINDOW)
    tr = (prep.panel["date"] < prep.split["test_start"]) & prep.panel["dc_power_kw"].notna()
    rows = [
        {"partition": "Site-days with reported load (modelled sites)", "count": int(prep.observed.sum())},
        {"partition": "Site-days filled by short-gap interpolation (inputs only, never scored)",
         "count": int((prep.present & ~prep.observed).sum())},
        {"partition": "Irregular site-days (any consensus outlier)",
         "count": int((prep.panel["irr_fc"] | prep.panel["irr_ae"]).sum())},
        {"partition": "  ... in forecasting telemetry (load, traffic, RAN energy)", "count": int(prep.fc_irr.sum())},
        {"partition": "  ... in health indicators (autoencoder features)", "count": int(prep.ae_irr.sum())},
        {"partition": "Forecasting partition: clean 35-day sequences (train)", "count": int(len(fc["train"]))},
        {"partition": "  sequences discarded as irregular (train)", "count": int(fc["n_train_all"] - len(fc["train"]))},
        {"partition": "Anomaly-detection partition: clean 14-day sequences (train)", "count": int(ae_clean.sum())},
        {"partition": "  sequences discarded as irregular (train)", "count": int((ae_all & ~ae_clean).sum())},
        {"partition": "Training-period irregular day rate",
         "count": float((prep.panel.loc[tr, "irr_fc"] | prep.panel.loc[tr, "irr_ae"]).mean())},
    ]
    t = pd.DataFrame(rows)
    t.to_csv(C.TAB_DIR / "partition_summary.csv", index=False)
    return t


def run(prep: Prepared, rep):
    panel = prep.panel
    prep.prep_log.to_csv(C.TAB_DIR / "preprocessing_log.csv", index=False)
    import shutil
    shutil.copy(C.CACHE_DIR / "placeholders.csv", C.TAB_DIR / "placeholder_replacement.csv")

    within, pooled = eda.correlations(panel)
    eda.plot_heatmap(within, "fig04_correlation_within_site.png",
                     f"Within-site correlation of {len(within)} candidate features (site means removed)")
    eda.plot_heatmap(pooled, "figS1_correlation_pooled.png",
                     f"Pooled correlation of {len(pooled)} candidate features (between- and within-site)")
    t1 = eda.table_one(within)
    print(t1.to_string())
    eda.plot_histograms(panel)
    z = panel[[c for c in panel.columns if c.startswith("z_")]]
    flags = panel[["out_" + c for c in F.SCREENED]].rename(columns=lambda c: c[4:])
    eda.plot_boxplots(z, flags, prep.split, panel)
    eda.plot_kde_screening(z, prep.split, panel)
    prep.screen_info.to_csv(C.TAB_DIR / "outlier_screening.csv", index=False)
    print(prep.screen_info.round(4).to_string())
    print(partition_table(prep).to_string())
    plot_site_signals(prep, rep)

    # Fleet overview by configuration
    st = prep.static.copy()
    st["mean_load_kw"] = prep.site_mean_kw
    ov = st.groupby(["power_config", "site_type"]).agg(sites=("mean_load_kw", "size"),
                                                       mean_load_kw=("mean_load_kw", "mean")).reset_index()
    ov.to_csv(C.TAB_DIR / "fleet_overview.csv", index=False)
    print(ov.to_string())
