"""Data acquisition, preprocessing and feature engineering (paper Sec. III-A/B).

Builds one tidy *site-day panel* from the four raw sources:

* DS01 rectifier / power-system daily KPIs  (the "Fluke power log" analogue)
* DS02 site locations and cell configuration (static site attributes)
* DS03 hardware inventory                    (board power-on events -> root-cause context)
* DS04 hourly regional weather               (exogenous drivers)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

# Raw rectifier column -> short analysis name. Columns not listed are dropped
# (all-empty, constant, or exact duplicates of a kept column; see PREP_LOG).
RENAME = {
    "SITECODE": "site",
    "Date": "date",
    "Site Type": "site_type",
    "Avg DC Load Current (A)": "dc_current_a",
    "Max DC Load Current (A)": "dc_current_max_a",
    "Min DC Load Current (A)": "dc_current_min_a",
    "Avg DC Load Power (kW)": "dc_power_kw",
    "Max DC Load Power (kW)": "dc_power_max_kw",
    "Min DC Load Power (kW)": "dc_power_min_kw",
    "DC Load Energy Consumption (kWh)": "dc_energy_kwh",
    "Total Energy Input (kWh)": "energy_in_kwh",
    "Mains Supply (kWh)": "mains_kwh",
    "Mains Supply Duration (h)": "mains_h",
    "Mains Outage Duration (h)": "mains_outage_h",
    "Mains Outage Times(h)": "mains_outage_n",
    "Longest Outage Duration (h)": "longest_outage_h",
    "D.G. Supply (kWh)": "dg_kwh",
    "D.G. Supply Duration (h)": "dg_h",
    "D.G. Start Times": "dg_starts",
    "D.G. Fuel Consumption(L)": "dg_fuel_l",
    "D.G. Abnormal Runtime (h)": "dg_abnormal_h",
    "Avg Fuel Consumption/kWh(L/kWh)": "fuel_per_kwh",
    "Solar Supply (kWh)": "solar_kwh",
    "Solar Supply Duration (h)": "solar_h",
    "Multi-energy Concurrent Supply Duration (h)": "multi_supply_h",
    "Battery Discharge Duration (h)": "batt_discharge_h",
    "BLVD Duration (h)": "blvd_h",
    "LLVD Duration (h)": "llvd_h",
    "Energy Consumption (kWh)": "aux_energy_kwh",
    "Avg Battery Temperature (℃)": "batt_temp_c",
    "Highest Battery Temperature (℃)": "batt_temp_max_c",
    "Avg Indoor Temperature (℃)": "indoor_temp_c",
    "Highest Indoor Temperature (℃)": "indoor_temp_max_c",
    "Avg Indoor Humidity (%)": "indoor_hum",
    "ENERGYBOARD": "ran_energy_kwh",
    "ENERGYLTE": "lte_energy_kwh",
    "ENERGYNR": "nr_energy_kwh",
    "TRAFFIC_GB": "traffic_gb",
}

# Human-readable labels used in every figure and table.
LABELS = {
    "dc_current_a": "DC Load Current Avg (A)",
    "dc_current_max_a": "DC Load Current Max (A)",
    "dc_current_min_a": "DC Load Current Min (A)",
    "dc_power_kw": "DC Load Power Avg (kW)",
    "dc_power_max_kw": "DC Load Power Max (kW)",
    "dc_power_min_kw": "DC Load Power Min (kW)",
    "dc_energy_kwh": "DC Load Energy (kWh)",
    "energy_in_kwh": "Total Energy Input (kWh)",
    "mains_kwh": "Mains Supply (kWh)",
    "mains_h": "Mains Supply Duration (h)",
    "mains_outage_h": "Mains Outage Duration (h)",
    "mains_outage_n": "Mains Outage Count",
    "longest_outage_h": "Longest Outage (h)",
    "dg_kwh": "DG Supply (kWh)",
    "dg_h": "DG Supply Duration (h)",
    "dg_starts": "DG Start Count",
    "dg_fuel_l": "DG Fuel (L)",
    "dg_abnormal_h": "DG Abnormal Runtime (h)",
    "fuel_per_kwh": "DG Fuel per kWh (L/kWh)",
    "solar_kwh": "Solar Supply (kWh)",
    "solar_h": "Solar Supply Duration (h)",
    "multi_supply_h": "Multi-source Supply (h)",
    "batt_discharge_h": "Battery Discharge (h)",
    "blvd_h": "BLVD Duration (h)",
    "llvd_h": "LLVD Duration (h)",
    "aux_energy_kwh": "Aux Energy Meter (kWh)",
    "batt_temp_c": "Battery Temp Avg (°C)",
    "batt_temp_max_c": "Battery Temp Max (°C)",
    "indoor_temp_c": "Indoor Temp Avg (°C)",
    "indoor_temp_max_c": "Indoor Temp Max (°C)",
    "indoor_hum": "Indoor Humidity Avg (%)",
    "ran_energy_kwh": "RAN Board Energy (kWh)",
    "lte_energy_kwh": "LTE Energy (kWh)",
    "nr_energy_kwh": "NR (5G) Energy (kWh)",
    "traffic_gb": "Traffic (GB)",
    "amb_temp_c": "Ambient Temp Avg (°C)",
    "amb_temp_max_c": "Ambient Temp Max (°C)",
    "amb_humidity": "Ambient Humidity (%)",
    "amb_wind_kph": "Wind Speed (km/h)",
    "amb_pressure_mb": "Pressure (mb)",
    "amb_uv_max": "UV Index Max",
    "dc_voltage_v": "DC Bus Voltage (V)",
    "conv_ratio": "Conversion Ratio DC/AC",
    "load_swing": "Load Swing Index",
    "dc_ran_ratio": "DC/RAN Energy Ratio",
}

# Additive (extensive) quantities are summed when a site reports two power
# systems on the same day; everything else is averaged.
_EXTENSIVE = {
    "dc_current_a", "dc_current_max_a", "dc_current_min_a", "dc_power_kw",
    "dc_power_max_kw", "dc_power_min_kw", "dc_energy_kwh", "energy_in_kwh",
    "mains_kwh", "dg_kwh", "dg_fuel_l", "solar_kwh", "aux_energy_kwh",
}

# Physical plausibility limits. Values outside are exporter placeholders or
# sensor glitches (e.g. 14,189 kW average load on a ~5 kW site) -> NaN.
# (low, high, low_inclusive_is_invalid)
_LIMITS = {
    "dc_power_kw": (0, 40, True),
    "dc_power_max_kw": (0, 60, True),
    "dc_power_min_kw": (0, 40, False),
    "dc_current_a": (0, 800, True),
    "dc_current_max_a": (0, 1200, True),
    "dc_current_min_a": (0, 800, False),
    "dc_energy_kwh": (0, 960, True),
    "energy_in_kwh": (0, 1200, False),
    "mains_kwh": (0, 1200, False),
    "dg_kwh": (0, 1200, False),
    "solar_kwh": (0, 1200, False),
    "dg_fuel_l": (0, 500, False),
    "fuel_per_kwh": (0, 5, False),
    "dg_starts": (0, 48, False),
    "mains_outage_n": (0, 96, False),
    "batt_temp_c": (0, 70, True),
    "batt_temp_max_c": (0, 80, True),
    "indoor_temp_c": (0, 70, True),
    "indoor_temp_max_c": (0, 80, True),
    "indoor_hum": (0, 100, True),
    "aux_energy_kwh": (0, 500, False),
}
_DURATIONS = ["mains_h", "mains_outage_h", "longest_outage_h", "dg_h", "dg_abnormal_h",
              "solar_h", "multi_supply_h", "batt_discharge_h", "blvd_h", "llvd_h"]


class PrepLog:
    """Collects a step-by-step record of the preprocessing (reported as a table)."""

    def __init__(self):
        self.rows = []
        self.placeholders = None

    def add(self, step, detail, rows=None):
        self.rows.append({"step": step, "detail": detail, "rows_after": rows})
        print(f"[prep] {step}: {detail}" + (f" (rows={rows:,})" if rows is not None else ""))

    def frame(self):
        return pd.DataFrame(self.rows)


# --------------------------------------------------------------------------
# Rectifier KPIs
# --------------------------------------------------------------------------
def load_rectifier(log: PrepLog) -> pd.DataFrame:
    first = pd.read_csv(C.DATA_DIR / C.RECTIFIER_FILES[0], low_memory=False)
    parts = [first] + [
        pd.read_csv(C.DATA_DIR / f, header=None, names=first.columns, low_memory=False)
        for f in C.RECTIFIER_FILES[1:]
    ]
    df = pd.concat(parts, ignore_index=True)
    log.add("load", f"3 export parts concatenated, {df.shape[1]} raw columns, "
                    f"{df['SITECODE'].nunique():,} sites", len(df))

    # Exports mix zero-padded and unpadded M/D/YYYY (Excel locale artefact);
    # file order confirms month-first for both spellings.
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    log.add("parse dates", f"mixed-padding M/D/YYYY -> {df['Date'].min().date()} .. "
                           f"{df['Date'].max().date()}", len(df))

    n0 = len(df)
    df = df.drop_duplicates()
    log.add("drop exact duplicates", f"{n0 - len(df):,} repeated export rows removed", len(df))

    for c in df.columns:
        if df[c].dtype == object and df[c].astype(str).str.endswith("%").any():
            df[c] = pd.to_numeric(df[c].astype(str).str.rstrip("%"), errors="coerce")
    log.add("percent strings", "'100%'-style text converted to numeric", len(df))

    dropped = [c for c in df.columns if c not in RENAME]
    df = df[list(RENAME)].rename(columns=RENAME)
    log.add("column pruning", f"{len(dropped)} columns dropped (all-empty, constant, or exact "
                              f"copies such as Site Fuel == DG Fuel); {len(RENAME) - 2} kept", len(df))

    # Three sites report two independent power systems per day -> one site-day.
    dup = df.duplicated(["site", "date"], keep=False)
    if dup.any():
        num = [c for c in df.columns if c not in ("site", "date", "site_type")]
        agg = {c: ("sum" if c in _EXTENSIVE else "mean") for c in num}
        agg["site_type"] = "first"
        merged = df[dup].groupby(["site", "date"], as_index=False).agg(agg)
        # sum() of an all-NaN group is 0; restore NaN
        cnt = df[dup].groupby(["site", "date"])[num].count().reset_index(drop=True)
        for c in num:
            merged.loc[cnt[c].values == 0, c] = np.nan
        n_sites = df.loc[dup, "site"].nunique()
        df = pd.concat([df[~dup], merged], ignore_index=True)
        log.add("merge dual power systems", f"{n_sites} sites with two rectifier systems "
                                            f"aggregated per day", len(df))
    return df.sort_values(["site", "date"]).reset_index(drop=True)


def clean_placeholders(df: pd.DataFrame, log: PrepLog) -> pd.DataFrame:
    df = df.copy()
    total = 0
    detail = []
    for c, (lo, hi, lo_invalid) in _LIMITS.items():
        bad = (df[c] > hi) | ((df[c] <= lo) if lo_invalid else (df[c] < lo))
        n = int(bad.sum())
        if n:
            df.loc[bad, c] = np.nan
            total += n
            detail.append(f"{c} outside limits:{n}")
    for c in _DURATIONS:
        bad = (df[c] < 0) | (df[c] > 24.5)
        if bad.any():
            df.loc[bad, c] = np.nan
            total += int(bad.sum())
            detail.append(f"{c} > 24 h:{int(bad.sum())}")

    # Implied DC bus voltage must sit in the 48 V-plant envelope (LLVD ~44 V,
    # boost charge ~57 V). Outside it, the power/current pair is inconsistent.
    v = df["dc_power_kw"] * 1000 / df["dc_current_a"]
    bad = (v < 40) | (v > 60)
    df.loc[bad, ["dc_power_kw", "dc_energy_kwh"]] = np.nan
    total += int(bad.sum())
    detail.append(f"P/I-inconsistent:{int(bad.sum())}")

    # NMS counters report 0 when the board export is missing while the
    # rectifier shows load -> placeholder, not a real zero.
    for c in ["ran_energy_kwh", "lte_energy_kwh", "nr_energy_kwh", "traffic_gb"]:
        bad = (df[c] <= 0) & (df["dc_power_kw"] > 0) if c in ("ran_energy_kwh", "traffic_gb") else df[c] < 0
        df.loc[bad, c] = np.nan
        total += int(bad.sum())
        detail.append(f"{c}=0:{int(bad.sum())}")
    log.add("placeholder replacement", f"{total:,} implausible values set to NaN "
                                       f"(physical limits, P/I consistency, NMS zero-placeholders)", len(df))
    log.placeholders = pd.DataFrame([d.split(":") for d in detail], columns=["rule", "values_replaced"])
    return df


def derive(df: pd.DataFrame) -> pd.DataFrame:
    """Derived health indicators (also re-applied to fault-injected rows)."""
    df["dc_voltage_v"] = df["dc_power_kw"] * 1000 / df["dc_current_a"]
    ein = df["energy_in_kwh"].where(df["energy_in_kwh"] >= 5)
    df["conv_ratio"] = (df["dc_energy_kwh"] / ein).where(lambda s: s.between(0.2, 3))
    df["load_swing"] = ((df["dc_power_max_kw"] - df["dc_power_min_kw"]) / df["dc_power_kw"]).where(
        lambda s: s.between(0, 5))
    ran = df["ran_energy_kwh"].where(df["ran_energy_kwh"] >= 2)
    df["dc_ran_ratio"] = (df["dc_energy_kwh"] / ran).where(lambda s: s.between(0.1, 10))
    return df


def engineer(df: pd.DataFrame, log: PrepLog) -> pd.DataFrame:
    df = derive(df.copy())
    log.add("feature engineering", "derived DC bus voltage (P/I), conversion ratio (DC out/AC in), "
                                   "load swing ((Pmax-Pmin)/Pavg), DC/RAN energy ratio", len(df))
    return df


# --------------------------------------------------------------------------
# Auxiliary sources
# --------------------------------------------------------------------------
def load_weather() -> pd.DataFrame:
    w = pd.read_csv(C.DATA_DIR / C.WEATHER_FILE)
    w["TIME"] = pd.to_datetime(w["TIME"], format="%m/%d/%Y %H:%M")  # local time (UTC+3)
    d = w.groupby(w["TIME"].dt.normalize()).agg(
        amb_temp_c=("TEMP_C", "mean"),
        amb_temp_max_c=("TEMP_C", "max"),
        amb_humidity=("HUMIDITY", "mean"),
        amb_wind_kph=("WIND_KPH", "mean"),
        amb_pressure_mb=("PRESSURE_MB", "mean"),
        amb_uv_max=("UV", "max"),
    )
    d.index.name = "date"
    return d


def load_site_static(rect: pd.DataFrame) -> pd.DataFrame:
    loc = pd.read_csv(C.DATA_DIR / C.LOCATION_FILE)
    s = loc.groupby("ANONYMIZED_SITECODE").agg(
        cells_2g=("CELLS2G", "first"), cells_4g=("CELLS4G", "first"),
        cells_5g=("CELLS5G", "first"), n_cells=("ANOM_CELLCODE", "size"),
        lat=("LAT", "first"), lon=("LON", "first"))
    s.index.name = "site"

    inv = pd.read_csv(C.DATA_DIR / C.INVENTORY_FILE, low_memory=False, na_values=["\\N"])
    radio = {"MRRU", "AIRU", "LRRU", "MRFU", "LRFU", "EPRRU", "WRFU", "MPRF", "RHUB", "BRU"}
    b = inv.groupby("CODE").agg(
        n_boards=("BOARDNAME", lambda x: int((x != "FModule").sum())),
        n_radio_units=("BOARDNAME", lambda x: int(x.isin(radio).sum())))
    b.index.name = "site"

    g = rect.groupby("site")
    cfg = pd.DataFrame({
        "site_type": g["site_type"].agg(lambda x: x.mode().iat[0]),
        "mains_share": g["mains_kwh"].apply(lambda x: x.notna().mean()),
        "dg_share": g["dg_kwh"].apply(lambda x: (x > 0).mean()),
        "solar_share": g["solar_kwh"].apply(lambda x: (x > 0).mean()),
    })
    cfg["power_config"] = np.select(
        [(cfg.mains_share > 0.5) & (cfg.dg_share <= 0.5),
         (cfg.dg_share > 0.5) & (cfg.solar_share > 0.5),
         (cfg.dg_share > 0.5)],
        ["Grid", "DG+Solar", "DG"], default="Other")
    return cfg.join(s, how="left").join(b, how="left")


def load_hw_events() -> pd.DataFrame:
    """Daily count of hardware boards first powered on, per site (root-cause context)."""
    inv = pd.read_csv(C.DATA_DIR / C.INVENTORY_FILE, low_memory=False, na_values=["\\N"],
                      usecols=["CODE", "BOARDNAME", "FIRSTPOWERONTIME"])
    inv = inv[inv["BOARDNAME"] != "FModule"]
    inv["date"] = pd.to_datetime(inv["FIRSTPOWERONTIME"], format="%m/%d/%Y", errors="coerce")
    ev = inv.dropna(subset=["date"]).groupby(["CODE", "date"]).size().rename("hw_poweron_n")
    ev.index.names = ["site", "date"]
    return ev.reset_index()


# --------------------------------------------------------------------------
# Panel
# --------------------------------------------------------------------------
def build_panel(force: bool = False):
    """Return (panel, static, prep_log). Cached under .cache/."""
    C.CACHE_DIR.mkdir(exist_ok=True)
    p_path, s_path, l_path = (C.CACHE_DIR / "panel.parquet", C.CACHE_DIR / "static.parquet",
                              C.CACHE_DIR / "prep_log.csv")
    if p_path.exists() and not force:
        return pd.read_parquet(p_path), pd.read_parquet(s_path), pd.read_csv(l_path)

    log = PrepLog()
    rect = load_rectifier(log)
    rect = clean_placeholders(rect, log)
    rect = engineer(rect, log)
    static = load_site_static(rect)

    # Complete daily calendar per site so that gaps are explicit.
    dates = pd.date_range(rect["date"].min(), rect["date"].max(), freq="D")
    full = pd.MultiIndex.from_product([static.index, dates], names=["site", "date"])
    n_obs = len(rect)
    panel = rect.drop(columns="site_type").set_index(["site", "date"]).reindex(full).reset_index()
    log.add("calendar reindex", f"{len(dates)} days x {len(static):,} sites; "
                                f"{1 - n_obs / len(panel):.1%} of site-days have no export", len(panel))

    w = load_weather()
    panel = panel.merge(w, left_on="date", right_index=True, how="left")
    log.add("join weather", "hourly regional weather -> daily mean/max temperature, humidity, "
                            "wind, pressure, UV", len(panel))

    ev = load_hw_events()
    panel = panel.merge(ev, on=["site", "date"], how="left")
    panel["hw_poweron_n"] = panel["hw_poweron_n"].fillna(0)
    log.add("join inventory events", f"{int(panel['hw_poweron_n'].sum()):,} board power-on events "
                                     f"inside the study window", len(panel))

    dow = panel["date"].dt.dayofweek
    panel["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    panel["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    panel["weekend"] = dow.isin([4, 5]).astype(float)  # Fri/Sat weekend (UTC+3 region)

    panel.to_parquet(p_path)
    static.to_parquet(s_path)
    lf = log.frame()
    lf.to_csv(l_path, index=False)
    log.placeholders.to_csv(C.CACHE_DIR / "placeholders.csv", index=False)
    return panel, static, lf


if __name__ == "__main__":
    p, s, l = build_panel(force=True)
    print(p.shape, s.shape)
    print(s["power_config"].value_counts(), s["site_type"].value_counts())
