"""Central configuration: paths, feature sets and model hyper-parameters.

The hyper-parameters mirror Table II of Metry et al. ("Hybrid Forecasting-Anomaly
Detection Approach for Smart Building Energy Monitoring") wherever the paper
specifies them, and are adapted only where the change of sampling rate
(sub-minute panel readings -> daily site KPIs) makes a literal copy meaningless.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT
CACHE_DIR = ROOT / ".cache"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"
MODEL_DIR = OUT_DIR / "models"

RECTIFIER_FILES = [
    "ANOM_DS01_Rectifier_Data Part 1.csv",
    "ANOM_DS01_Rectifier_Data Part 2.csv",
    "ANOM_DS01_Rectifier_Data Part 3.csv",
]
LOCATION_FILE = "ANOM_DS02_SiteLocations.csv"
INVENTORY_FILE = "ANOM_DS03_SiteInventory_in_.csv"
WEATHER_FILE = "ANOM_DS04_WeatherData.csv"

SEED = 42

# --------------------------------------------------------------------------
# Sites / periods
# --------------------------------------------------------------------------
MIN_VALID_DAYS = 300          # a site needs this much usable history to be modelled
MAX_INTERP_GAP = 3            # days; short telemetry gaps are interpolated
TRAIN_FRACTION = 0.80         # chronological 80/20 split (paper Sec. III-F)
VAL_FRACTION_OF_TRAIN = 0.10  # tail of the training period held out for model selection / thresholds

# --------------------------------------------------------------------------
# Feature sets (chosen in the correlation analysis, Sec. III-C / Table I)
# --------------------------------------------------------------------------
TARGET = "dc_power_kw"

# Five forecasting inputs (paper: Vrms, Current, Frequency, PF, Active Power)
FORECAST_FEATURES = [
    "dc_power_kw",        # target history (Active Power analogue)
    "amb_temp_c",         # ambient temperature  (cooling load driver)
    "amb_humidity",       # ambient humidity
    "traffic_gb",         # carried traffic       (radio load driver)
    "ran_energy_kwh",     # RAN board energy from the NMS (independent meter)
]
# Calendar encodings are appended to the forecasting inputs (known in advance).
CALENDAR_FEATURES = ["dow_sin", "dow_cos", "weekend"]

# Six autoencoder inputs (paper: THD V, THD A, Unbalance, Harmonic pollution,
# V-harmonic 5, A-harmonic 5) -> power-system health / quality indicators.
AE_FEATURES = [
    "conv_ratio",         # DC output / AC input energy        (power-factor / efficiency analogue)
    "load_swing",         # (Pmax - Pmin) / Pavg                (current-distortion analogue)
    "dc_voltage_v",       # implied DC bus voltage  P / I       (voltage-quality analogue)
    "batt_discharge_h",   # battery discharge hours             (supply-quality indicator)
    "batt_temp_c",        # average battery temperature         (thermal stress)
    "dc_ran_ratio",       # DC load energy / RAN board energy   (meter-consistency / unbalance analogue)
]

SELECTED_FEATURES = FORECAST_FEATURES + AE_FEATURES  # the "11 selected features"

# --------------------------------------------------------------------------
# Forecasting model (paper Table II, adapted)
# --------------------------------------------------------------------------
FC_WINDOW = 28        # paper: 60 time steps; here 28 days = four weekly cycles
FC_HORIZON = 7        # P(t+1) ... P(t+h) as in Algorithm 1; headline metrics on t+1
FC_HIDDEN = (64, 32)  # two stacked LSTM layers, dropout after each
FC_DROPOUT = 0.2
FC_BATCH = 64
FC_LR = 1e-3
FC_EPOCHS = 20

# --------------------------------------------------------------------------
# LSTM autoencoder
# --------------------------------------------------------------------------
AE_WINDOW = 14
AE_HIDDEN = 32
AE_LATENT = 16
AE_DROPOUT = 0.2
AE_BATCH = 64
AE_LR = 1e-3
AE_EPOCHS = 20

# --------------------------------------------------------------------------
# Outlier screening / thresholds
# --------------------------------------------------------------------------
IQR_K = 1.5                 # Tukey box-plot fences (per site)
KDE_TAIL_MASS = 0.01        # pooled-KDE low-density region holding 1 % of the mass
THRESHOLD_PERCENTILE = 95   # KDE-derived dynamic threshold (paper Sec. IV-B)
