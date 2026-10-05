# Forecasting and Anomaly Detection for Telecom Site Energy

A re-application of **Metry et al., "Hybrid Forecasting-Anomaly Detection Approach for Smart
Building Energy Monitoring"** to the raw power-system data of a telecom site fleet:
an LSTM forecaster for DC load, an LSTM autoencoder for power-system health, KDE-derived
thresholds, OR/AND fusion, and indicator-based root-cause analysis.

* **[REPORT.md](REPORT.md)**: the study (method, results, comparison with the paper, limitations).
* **[outputs/RESULTS.md](outputs/RESULTS.md)**: every result table (auto-generated).
* `outputs/figures/`: all figures. `outputs/tables/`: all tables as CSV/JSON. `outputs/models/`: trained weights.

## Data (repository root)

| File | Content |
|---|---|
| `ANOM_DS01_Rectifier_Data Part 1-3.csv` | Daily rectifier / power-system KPIs per site (+ RAN energy and traffic counters) |
| `ANOM_DS02_SiteLocations.csv` | Cells, technologies and anonymised coordinates per site |
| `ANOM_DS03_SiteInventory_in_.csv` | Hardware inventory with board power-on dates |
| `ANOM_DS04_WeatherData.csv` | Hourly regional weather |

## Running

```bash
pip install -r requirements.txt
python run_study.py                       # full pipeline, about 1 h on 4 CPU cores
python run_study.py hybrid validate report   # re-run later stages from cached results
python run_study.py sensitivity           # look-back window ablation (14 / 28 / 60 days)
```

Stages: `prepare` → `eda` → `forecast` → `autoencoder` → `hybrid` → `validate` → `report`.
`forecast-eval` / `autoencoder-eval` re-score the saved weights without retraining.
Intermediate data are cached in `.cache/` (git-ignored). All randomness is seeded.

## Code map

| Module | Paper section |
|---|---|
| `study/data.py` | III-A/B: loading, cleaning, placeholder replacement, feature engineering, data integration |
| `study/features.py` | III-E: site-relative scaling, KDE + box-plot outlier screening |
| `study/eda.py`, `study/stage_eda.py` | III-C/D/E: correlation heatmap, Table I, histograms, box plots, partition |
| `study/prepare.py` | Tensors, validity masks, clean/irregular partition |
| `study/models.py` | LSTM forecaster, LSTM autoencoder, training loop |
| `study/forecasting.py` | III-F, IV-A: forecasting, baselines, metrics (Table III), window sensitivity |
| `study/anomaly.py` | III-G, IV-B: autoencoder scoring, KDE threshold (Table IV) |
| `study/hybrid.py` | Algorithm 1, IV-C: OR/AND fusion, episodes, root-cause analysis |
| `study/validation.py` | Beyond the paper: synthetic fault injection, operational-event validation |
| `study/summary.py` | Results digest |
