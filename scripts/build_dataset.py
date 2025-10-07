import os
import pandas as pd
import numpy as np

# --------------------------------------------
# paths
# --------------------------------------------
DATA_DIR = "data"
OUT_DIR = "output"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# expected input files (drop your CSVs in /data)
# - owid_co2.csv            -> columns: iso_code,country,year,co2_per_capita
# - owid_pm25.csv           -> columns: iso_code,country,year,pm25
# - owid_life_expectancy.csv-> columns: iso_code,country,year,life_expectancy
# - forest_loss.csv         -> columns: iso_code,country,year,forest_loss_pct   (optional)
# - who_chronic.csv         -> columns: iso_code,country,year,chronic_disease_rate (optional)

def load_csv_safe(path, required_cols):
    if not os.path.exists(path):
        print(f"⚠️ missing file: {path} (skipping)")
        return pd.DataFrame(columns=required_cols)
    df = pd.read_csv(path)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return df[required_cols].copy()

def minmax_norm(x):
    x = x.astype(float)
    minv, maxv = np.nanmin(x), np.nanmax(x)
    if np.isnan(minv) or np.isnan(maxv) or maxv == minv:
        return pd.Series([np.nan] * len(x), index=x.index)
    return (x - minv) / (maxv - minv)

# --------------------------------------------
# load
# --------------------------------------------
co2 = load_csv_safe(
    os.path.join(DATA_DIR, "owid_co2.csv"),
    ["iso_code", "country", "year", "co2_per_capita"]
)

pm25 = load_csv_safe(
    os.path.join(DATA_DIR, "owid_pm25.csv"),
    ["iso_code", "country", "year", "pm25"]
)

life = load_csv_safe(
    os.path.join(DATA_DIR, "owid_life_expectancy.csv"),
    ["iso_code", "country", "year", "life_expectancy"]
)

forest = load_csv_safe(
    os.path.join(DATA_DIR, "forest_loss.csv"),
    ["iso_code", "country", "year", "forest_loss_pct"]
)

chronic = load_csv_safe(
    os.path.join(DATA_DIR, "who_chronic.csv"),
    ["iso_code", "country", "year", "chronic_disease_rate"]
)

# --------------------------------------------
# merge
# --------------------------------------------
dfs = [life, co2, pm25, forest, chronic]
base = None
for d in dfs:
    base = d if base is None else base.merge(d, on=["iso_code", "country", "year"], how="outer")

# keep only country-level iso codes (drop aggregates like OWID_* if present)
base = base[~base["iso_code"].astype(str).str.startswith("OWID")]

# --------------------------------------------
# simple per-year normalization & eco-health score
# --------------------------------------------
def compute_yearly_scores(g):
    # normalize each available feature
    co2_n     = 1 - minmax_norm(g["co2_per_capita"])  # lower CO2 is "better"
    pm25_n    = 1 - minmax_norm(g["pm25"])            # lower PM2.5 is "better"
    forest_n  = 1 - minmax_norm(g["forest_loss_pct"]) # lower loss is "better"
    life_n    = minmax_norm(g["life_expectancy"])     # higher life expectancy is "better"
    chronic_n = 1 - minmax_norm(g["chronic_disease_rate"])  # lower chronic rate is "better"

    # combine (ignore NaNs by taking mean of available components)
    components = pd.concat([co2_n, pm25_n, forest_n, life_n, chronic_n], axis=1)
    eco_score = components.mean(axis=1, skipna=True)

    out = g.copy()
    out["eco_health_score"] = eco_score
    return out

panel = base.groupby("year", group_keys=False).apply(compute_yearly_scores)

# keep a clean set of columns
keep_cols = [
    "iso_code","country","year",
    "co2_per_capita","pm25","forest_loss_pct",
    "life_expectancy","chronic_disease_rate","eco_health_score"
]
for c in keep_cols:
    if c not in panel.columns:
        panel[c] = np.nan

panel = panel[keep_cols].sort_values(["country","year"]).reset_index(drop=True)

# save
out_path = os.path.join(OUT_DIR, "eco_health_panel.csv")
panel.to_csv(out_path, index=False)
print(f"✅ saved panel to {out_path} with shape {panel.shape}")
