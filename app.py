import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

st.set_page_config(page_title="healthy planet, healthy people", layout="wide")

@st.cache_data
def load_panel():
    df = pd.read_csv("output/eco_health_panel.csv")
    # basic cleaning
    df = df.dropna(subset=["iso_code","country","year"])
    df["year"] = df["year"].astype(int)
    return df

df = load_panel()

st.title("🌍 healthy planet, healthy people")
st.caption("data as empathy — exploring the link between environment and human health")

# -----------------------------
# sidebar controls
# -----------------------------
years = sorted(df["year"].dropna().unique())
min_year, max_year = int(min(years)), int(max(years))
year = st.sidebar.slider("year", min_year, max_year, max_year, step=1)

metric_opts = {
    "eco-health score": "eco_health_score",
    "life expectancy": "life_expectancy",
    "pm2.5 (μg/m³)": "pm25",
    "co₂ per capita (t)": "co2_per_capita",
    "forest loss (%)": "forest_loss_pct",
    "chronic disease rate": "chronic_disease_rate",
}
map_metric_label = st.sidebar.selectbox("choropleth metric", list(metric_opts.keys()), index=0)
map_metric = metric_opts[map_metric_label]

focus_countries = st.sidebar.multiselect(
    "highlight countries (optional)",
    sorted(df["country"].unique())[:300],
)

# -----------------------------
# choropleth map
# -----------------------------
st.subheader(f"🗺️ global map — {map_metric_label} ({year})")

df_y = df[df["year"] == year].copy()
fig_map = px.choropleth(
    df_y,
    locations="iso_code",
    color=map_metric,
    hover_name="country",
    color_continuous_scale="Viridis",
    title=None,
)
fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0))
st.plotly_chart(fig_map, use_container_width=True)

# -----------------------------
# dual-axis: co2 vs life expectancy (global avg)
# -----------------------------
st.subheader("📈 co₂ vs life expectancy (global average over time)")

global_trend = (
    df.groupby("year")
      .agg(co2=("co2_per_capita","mean"), life=("life_expectancy","mean"))
      .reset_index()
)

fig_dual = go.Figure()
fig_dual.add_trace(go.Scatter(x=global_trend["year"], y=global_trend["co2"],
                              name="co₂ per capita (t)", mode="lines+markers", yaxis="y1"))
fig_dual.add_trace(go.Scatter(x=global_trend["year"], y=global_trend["life"],
                              name="life expectancy (years)", mode="lines+markers", yaxis="y2"))

fig_dual.update_layout(
    xaxis=dict(title="year"),
    yaxis=dict(title="co₂ per capita (t)"),
    yaxis2=dict(title="life expectancy (years)", overlaying="y", side="right"),
    legend=dict(orientation="h"),
    margin=dict(l=0, r=0, t=0, b=0),
)
st.plotly_chart(fig_dual, use_container_width=True)

# -----------------------------
# correlation heatmap (selected year)
# -----------------------------
st.subheader(f"🔥 correlations between environment & health ({year})")

corr_cols = ["co2_per_capita","pm25","forest_loss_pct","life_expectancy","chronic_disease_rate","eco_health_score"]
corr_df = df_y[corr_cols].copy()
corr = corr_df.corr()

fig_corr = px.imshow(
    corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1
)
fig_corr.update_layout(margin=dict(l=0, r=0, t=0, b=0))
st.plotly_chart(fig_corr, use_container_width=True)

# -----------------------------
# feature importance: predict life expectancy
# -----------------------------
st.subheader("🧠 which factors explain life expectancy? (random forest)")

train = df.dropna(subset=["life_expectancy"])
X_cols = ["co2_per_capita","pm25","forest_loss_pct","chronic_disease_rate","eco_health_score"]
X = train[X_cols].fillna(train[X_cols].median())
y = train["life_expectancy"].values

if len(train) > 50:
    rf = RandomForestRegressor(n_estimators=300, random_state=42)
    rf.fit(X, y)
    yhat = rf.predict(X)
    r2 = r2_score(y, yhat)
    importances = pd.Series(rf.feature_importances_, index=X_cols).sort_values()

    col1, col2 = st.columns([1,1])
    with col1:
        st.metric("in-sample R²", f"{r2:.2f}")
        st.caption("quick diagnostic — for exploration only")

    with col2:
        fig_imp = px.bar(importances, orientation="h", title="feature importance")
        fig_imp.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_imp, use_container_width=True)
else:
    st.info("not enough rows to train a model yet — add more data to /data and rebuild.")

# -----------------------------
# country detail (optional highlight)
# -----------------------------
if focus_countries:
    st.subheader("🔍 country detail")
    for c in focus_countries:
        sub = df[df["country"] == c].sort_values("year")
        if sub.empty: 
            continue
        fig_c = px.line(
            sub, x="year", y=["eco_health_score","life_expectancy","pm25","co2_per_capita"],
            title=c, markers=True
        )
        fig_c.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_c, use_container_width=True)

st.caption("built with pandas • plotly • streamlit")
