import os
import pandas as pd
import numpy as np
import warnings

# --------------------------------------------
# paths
# --------------------------------------------
DATA_DIR = "data"
OUT_DIR = "output"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------
# preprocess Lily's original CSVs
# ------------------------------------------------
def preprocess_input_files():
    co2_path = os.path.join(DATA_DIR, "co2-emissions-per-capita.csv")
    life_path = os.path.join(DATA_DIR, "life-expectancy.csv")
    air_path = os.path.join(DATA_DIR, "long-run-air-pollution.csv")
    gdp_path = os.path.join(DATA_DIR, "United-States-GDP-Per-Capita-GDP-Per-Capita-US-2025-10-07-01-07.csv")

    # --- CO2 ---
    if os.path.exists(co2_path):
        co2 = pd.read_csv(co2_path)
        co2 = co2.rename(columns={
            "Entity": "country",
            "Year": "year",
            "Annual CO₂ emissions (per capita)": "co2_per_capita"
        })
        co2["year"] = pd.to_numeric(co2["year"], errors="coerce")  
        co2["iso_code"] = co2["country"].str[:3].str.upper()
        co2.to_csv(os.path.join(DATA_DIR, "owid_co2.csv"), index=False)

    # --- Life expectancy ---
    if os.path.exists(life_path):
        life = pd.read_csv(life_path)
        life = life.rename(columns={
            "Entity": "country",
            "Code": "iso_code",
            "Year": "year",
            "Period life expectancy at birth": "life_expectancy"
        })
        life["year"] = pd.to_numeric(life["year"], errors="coerce")  
        life.to_csv(os.path.join(DATA_DIR, "owid_life_expectancy.csv"), index=False)

    # --- Air pollution (proxy for PM2.5) ---
    if os.path.exists(air_path):
        air = pd.read_csv(air_path)
        air = air.rename(columns={
            "Entity": "country",
            "Code": "iso_code",
            "Year": "year",
            "Nitrogen oxides emissions from all sectors": "pm25"
        })
        air["year"] = pd.to_numeric(air["year"], errors="coerce")   
        air.to_csv(os.path.join(DATA_DIR, "owid_pm25.csv"), index=False)

    # --- GDP (optional add-on) ---
    if os.path.exists(gdp_path):
        gdp = pd.read_csv(gdp_path)
        # normalize column names to lowercase
        gdp.columns = [c.strip().lower() for c in gdp.columns]
        print(f"🧾 GDP columns detected: {gdp.columns.tolist()}")

        # find any column that looks like 'year' or 'date'
        year_col = next((c for c in gdp.columns if "year" in c or "date" in c), None)
        gdp_col  = next((c for c in gdp.columns if "gdp" in c and "capita" in c), None)

        if not year_col or not gdp_col:
            print(f"⚠️ skipping GDP: missing expected columns (found {gdp.columns.tolist()})")
        else:
            gdp = gdp.rename(columns={year_col: "year", gdp_col: "gdp_per_capita"})
            gdp["year"] = pd.to_numeric(gdp["year"], errors="coerce")
            gdp["country"] = "United States"
            gdp["iso_code"] = "USA"
            gdp.to_csv(os.path.join(DATA_DIR, "gdp_per_capita.csv"), index=False)
            print("✅ Processed GDP dataset successfully.")

# ✅ Call preprocessing
preprocess_input_files()

# --------------------------------------------
# helper functions
# --------------------------------------------
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
    # clean normalization with warning suppression
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        x = x.astype(float)
        if x.isna().all():
            return pd.Series([np.nan] * len(x), index=x.index)
        minv, maxv = np.nanmin(x), np.nanmax(x)
        if np.isnan(minv) or np.isnan(maxv) or maxv == minv:
            return pd.Series([np.nan] * len(x), index=x.index)
        return (x - minv) / (maxv - minv)

# --------------------------------------------
# load cleaned datasets
# --------------------------------------------
co2 = load_csv_safe(os.path.join(DATA_DIR, "owid_co2.csv"),
                    ["iso_code", "country", "year", "co2_per_capita"])
pm25 = load_csv_safe(os.path.join(DATA_DIR, "owid_pm25.csv"),
                     ["iso_code", "country", "year", "pm25"])
life = load_csv_safe(os.path.join(DATA_DIR, "owid_life_expectancy.csv"),
                     ["iso_code", "country", "year", "life_expectancy"])
forest = load_csv_safe(os.path.join(DATA_DIR, "forest_loss.csv"),
                       ["iso_code", "country", "year", "forest_loss_pct"])
chronic = load_csv_safe(os.path.join(DATA_DIR, "who_chronic.csv"),
                        ["iso_code", "country", "year", "chronic_disease_rate"])

# --------------------------------------------
# merge into one dataset
# --------------------------------------------
dfs = [life, co2, pm25, forest, chronic]
# keep only non-empty dataframes
dfs = [d for d in dfs if not d.empty]

if not dfs:
    raise ValueError("❌ No valid datasets found in /data folder.")

print(f"✅ merging {len(dfs)} datasets...")

base = dfs[0]
for d in dfs[1:]:
    common = list(set(base.columns) & set(d.columns))
    if "year" in common:
        base = base.merge(d, on=["iso_code", "country", "year"], how="outer")
    else:
        print(f"⚠️ skipping merge for dataset missing 'year' column: {d.columns.tolist()}")

# ensure year exists
if "year" not in base.columns:
    base["year"] = np.nan

base = base[~base["iso_code"].astype(str).str.startswith("OWID")]
# ✅ Ensure 'year' column exists and is numeric
if "year" not in base.columns:
    base["year"] = np.nan

# Some OWID CSVs label it differently or load as string
base.columns = [c.strip().lower() for c in base.columns]
if "year" in base.columns:
    base["year"] = pd.to_numeric(base["year"], errors="coerce")

# Drop rows missing a valid year (e.g., blank header rows)
base = base.dropna(subset=["year"])
base["year"] = base["year"].astype(int)
print(f"✅ Valid years detected: {base['year'].min()}–{base['year'].max()}")

print(f"📂 available columns before scoring: {base.columns.tolist()}")

# --------------------------------------------
# compute normalized eco-health score
# --------------------------------------------
def compute_yearly_scores(g):
    # handle missing columns gracefully
    def safe_norm(colname, invert=False):
        if colname not in g.columns:
            return pd.Series(np.nan, index=g.index)
        normed = minmax_norm(g[colname])
        return (1 - normed) if invert else normed

    co2_n     = safe_norm("co2_per_capita", invert=True)
    pm25_n    = safe_norm("pm25", invert=True)
    forest_n  = safe_norm("forest_loss_pct", invert=True)
    life_n    = safe_norm("life_expectancy")
    chronic_n = safe_norm("chronic_disease_rate", invert=True)

    components = pd.concat([co2_n, pm25_n, forest_n, life_n, chronic_n], axis=1)
    eco_score = components.mean(axis=1, skipna=True)

    out = g.copy()
    out["eco_health_score"] = eco_score
    return out

# --------------------------------------------
# safe groupby apply (no warnings)
# --------------------------------------------
# ✅ Make sure 'year' is numeric and non-null before grouping
base = base.dropna(subset=["year"])
base["year"] = base["year"].astype(int)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    panel = (
        base
        .groupby("year", group_keys=True)  # ✅ keep group keys
        .apply(lambda g: compute_yearly_scores(g))
        .reset_index(drop=True)
    )

# ------------------------------------------------
# data sanity checks before save
# ------------------------------------------------
print("\n🔍 Checking data coverage before export...")
print(panel.isna().mean().sort_values(ascending=False).head(10))

# keep only rows that have at least 2 key indicators
panel = panel.dropna(subset=["eco_health_score"], how="all")
panel = panel[panel[["life_expectancy","co2_per_capita","pm25"]].notna().sum(axis=1) >= 2]

# optional smoothing: forward-fill by country
# optional smoothing: forward-fill by country (safe if year missing)
if "year" not in panel.columns:
    panel["year"] = np.nan
if "country" not in panel.columns:
    panel["country"] = "Unknown"

panel = panel.copy()
if not panel.empty:
    try:
        panel = (
            panel.sort_values(["country", "year"])
                  .groupby("country", group_keys=False)
                  .apply(lambda g: g.ffill().bfill())
                  .reset_index(drop=True)
        )
    except KeyError as e:
        print(f"⚠️ Skipping sort/fill due to missing key: {e}")

# clip eco_score to [0,1]
panel["eco_health_score"] = panel["eco_health_score"].clip(0,1)

print(f"📊 after cleaning: {len(panel)} rows, {panel['year'].nunique()} years")

# --------------------------------------------
# summary info
# --------------------------------------------
countries = panel["country"].nunique()
years = panel["year"].nunique()
print(f"🌍 Dataset covers {countries} countries across {years} years "
      f"({int(panel['year'].min())}–{int(panel['year'].max())})")

# --------------------------------------------
# organize and save
# --------------------------------------------
keep_cols = [
    "iso_code", "country", "year",
    "co2_per_capita", "pm25", "forest_loss_pct",
    "life_expectancy", "chronic_disease_rate", "eco_health_score"
]

for c in keep_cols:
    if c not in panel.columns:
        panel[c] = np.nan

for c in ["country", "year"]:
    if c not in panel.columns:
        panel[c] = np.nan

panel = panel[keep_cols]
if panel["year"].notna().any():
    panel = panel.sort_values(["country", "year"])
panel = panel.reset_index(drop=True)
panel = panel.dropna(subset=["life_expectancy", "co2_per_capita", "pm25"], how="all")

panel = panel.query("year >= 1960 and year <= 2023")
out_path = os.path.join(OUT_DIR, "eco_health_panel.csv")
panel.to_csv(out_path, index=False)
print(f"✅ saved panel to {out_path} with shape {panel.shape}")
