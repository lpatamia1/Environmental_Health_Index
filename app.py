import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# ---------------------------------------------
# 🌍 PAGE CONFIG
# ---------------------------------------------
st.set_page_config(
    page_title="Global Eco-Health Dashboard: Connecting Environmental and Human Well-Being",
    layout="wide",
    page_icon="🌎"
)

# ---------------------------------------------
# 📦 LOAD DATA
# ---------------------------------------------
@st.cache_data
def load_panel():
    df = pd.read_csv("output/eco_health_panel.csv")
    df = df.dropna(subset=["iso_code", "country", "year"])
    df["year"] = df["year"].astype(int)
    return df

df = load_panel()

# ---------------------------------------------
# 🧭 HEADER
# ---------------------------------------------
st.title("Project Eco-Health: Global Dashboard Ov")
st.caption("Data as empathy — revealing how our planet’s well-being reflects our own.")

with st.expander("Project Overview"):
    st.markdown("""
    Humanity’s health and the planet’s health are deeply intertwined.  
    This dashboard visualizes the global connections between **environmental conditions**
    and **human well-being**, emphasizing how sustainability, air quality, and resource use
    shape life expectancy and overall quality of life.

    ### What You’ll Find Here
    - **CO₂ emissions**: A measure of industrial activity and environmental pressure.  
    - **PM2.5 levels**: Fine particulate pollution that directly affects respiratory and heart health.  
    - **Life expectancy**: A reflection of medical access, lifestyle, and environmental exposure.  
    - **Eco-Health Score**: A composite indicator summarizing environmental well-being.

    ### How to Explore
    - Use the **sidebar** to choose a **year** and a **metric** for the global map.  
    - Highlight one or more **countries** to see detailed trends across time.  
    - Switch between tabs for:
        - **Global Overview:** Geographic patterns and comparisons.  
        - **Trends:** How CO₂ and life expectancy evolve globally.  
        - **Correlations:** Relationships between environment and health.  
        - **Machine Learning:** Feature importance driving health outcomes.

    ### Why It Matters
    By visualizing these patterns, we can better understand where environmental
    inequality overlaps with health inequality — and how improving one can uplift the other.

    *Data sources include [Our World in Data](https://ourworldindata.org), [World Bank](https://data.worldbank.org), and [WHO](https://www.who.int/data).*
    """)

# ---------------------------------------------
# 🧭 TABS LAYOUT
# ---------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Global Overview",
    "Global Trends",
    "Correlations",
    "Machine Learning Insights",
    "Country Detail View"
])

# ensure output folder for visuals
os.makedirs("output/charts", exist_ok=True)

# ---------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------
st.sidebar.markdown("## Dashboard Controls")

# ---- Dataset Summary ----
st.sidebar.markdown("### Dataset Overview")
st.sidebar.markdown(f"""
**Countries:** `{df['country'].nunique()}`  
**Years:** `{df['year'].min()} – {df['year'].max()}`  
**Records:** `{len(df):,}`  

**Metrics Tracked:**  
- CO₂ Emissions  
- PM2.5 (Air Quality)  
- Life Expectancy  
- Eco-Health Score
""")

st.sidebar.markdown("---")

# ---- Interactive Controls ----
st.sidebar.markdown("### Explore Data")

years = sorted(df["year"].dropna().unique())
min_year, max_year = int(min(years)), int(max(years))
year = st.sidebar.slider("Select Year", min_year, max_year, max_year, step=1)

metric_opts = {
    "Eco-Health Score": "eco_health_score",
    "Life Expectancy": "life_expectancy",
    "PM2.5 (μg/m³)": "pm25",
    "CO₂ per Capita (tons)": "co2_per_capita",
}
map_metric_label = st.sidebar.selectbox("Choropleth Metric", list(metric_opts.keys()), index=0)
map_metric = metric_opts[map_metric_label]

focus_countries = st.sidebar.multiselect(
    "Highlight Countries",
    sorted(df["country"].unique())[:300],
)

st.sidebar.markdown("---")
st.sidebar.caption("Use the tabs above to explore global patterns, correlations, and analytical insights.")

# ---------------------------------------------
# GLOBAL MAP
# ---------------------------------------------
with tab1:
    st.subheader(f"Global Map — {map_metric_label} ({year})")

    df_y = df[df["year"] == year].copy()
    fig_map = px.choropleth(
        df_y,
        locations="iso_code",
        color=map_metric,
        hover_name="country",
        color_continuous_scale="Blues",
    )
    fig_map.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar_title=map_metric_label,
    )
    st.plotly_chart(fig_map, use_container_width=True)

    top5 = df_y.nlargest(5, map_metric)[["country", map_metric]]
    bottom5 = df_y.nsmallest(5, map_metric)[["country", map_metric]]
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top 5 Countries**")
        st.table(top5)
    with col2:
        st.markdown("**Bottom 5 Countries**")
        st.table(bottom5)

    st.markdown(f"""
    **Analysis:**  
    The global map for **{map_metric_label.lower()}** in **{year}** highlights disparities across regions.  
    Lighter shades indicate *lower* values, while darker blues indicate *higher impact*.
    """)

# ---------------------------------------------
# CO₂ VS LIFE EXPECTANCY (GLOBAL TREND)
# ---------------------------------------------
with tab2:
    st.subheader("CO₂ vs Life Expectancy — Global Trend")

    global_trend = (
        df.groupby("year")
          .agg(co2=("co2_per_capita", "mean"),
               life=("life_expectancy", "mean"))
          .reset_index()
    )

    fig_dual = go.Figure()
    fig_dual.add_trace(go.Scatter(
        x=global_trend["year"], y=global_trend["co2"],
        name="CO₂ per Capita (tons)", mode="lines+markers",
        line=dict(color="#4682B4", width=3)
    ))
    fig_dual.add_trace(go.Scatter(
        x=global_trend["year"], y=global_trend["life"],
        name="Life Expectancy (years)", mode="lines+markers",
        line=dict(color="#1E90FF", width=3), yaxis="y2"
    ))

    fig_dual.update_layout(
        xaxis=dict(title="Year"),
        yaxis=dict(title="CO₂ per Capita (tons)"),
        yaxis2=dict(title="Life Expectancy (years)", overlaying="y", side="right"),
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_dual, use_container_width=True)

    st.markdown("""
    **Analysis:**  
    CO₂ emissions and life expectancy both rise over time — but this hides inequality.  
    High-income countries live longer despite higher emissions, revealing uneven progress.
    """)

# ---------------------------------------------
# CORRELATION HEATMAP
# ---------------------------------------------
with tab3:
    st.subheader(f"Correlations Between Environment & Health ({year})")

    corr_cols = ["co2_per_capita", "pm25", "forest_loss_pct",
                 "life_expectancy", "chronic_disease_rate", "eco_health_score"]
    df_y = df[df["year"] == year]
    corr = df_y[corr_cols].corr()

    fig_corr = px.imshow(
        corr, text_auto=".2f",
        color_continuous_scale="ice_r",
        zmin=-1, zmax=1,
    )
    fig_corr.update_layout(
        margin=dict(l=0, r=0, t=20, b=0),
        coloraxis_colorbar_title="Correlation",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("""
    **Analysis:**  
    - Strong *negative* correlations show that higher CO₂ or PM2.5 reduce life expectancy.  
    - Positive links with the eco-health score confirm the tie between environmental and human health.
    """)

# ---------------------------------------------
# FEATURE IMPORTANCE (ML MODEL)
# ---------------------------------------------
with tab4:
    st.subheader("🧠 Key Drivers of Life Expectancy")

    train = df.dropna(subset=["life_expectancy"])
    X_cols = ["co2_per_capita", "pm25", "forest_loss_pct",
              "chronic_disease_rate", "eco_health_score"]
    X = train[X_cols].fillna(train[X_cols].median())
    y = train["life_expectancy"].values

    if len(train) > 50:
        rf = RandomForestRegressor(n_estimators=300, random_state=42)
        rf.fit(X, y)
        yhat = rf.predict(X)
        r2 = r2_score(y, yhat)
        importances = pd.Series(rf.feature_importances_, index=X_cols).sort_values()

        col1, col2 = st.columns([1, 1])
        with col1:
            st.metric("Model R²", f"{r2:.2f}")
        with col2:
            fig_imp = px.bar(importances, orientation="h", color=importances, color_continuous_scale="Blues")
            st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown("""
        **Analysis:**  
        Eco-health score and air quality (PM2.5) are the top predictors of life expectancy —  
        showing that healthier ecosystems foster longer lives.
        """)

# ---------------------------------------------
# COUNTRY DETAIL VIEW
# ---------------------------------------------
with tab5:
    st.subheader("Country Detail Trends")
    st.caption("Compare multiple countries' environmental and health changes over time.")

    if not focus_countries:
        st.info("Select one or more countries from the sidebar to view their time-series trends.")
    else:
        for c in focus_countries:
            sub = df[df["country"] == c].sort_values("year")
            if sub.empty:
                continue
            fig_c = px.line(
                sub, x="year",
                y=["eco_health_score", "life_expectancy", "pm25", "co2_per_capita"],
                title=c,
                markers=True,
                color_discrete_sequence=px.colors.sequential.Blues
            )
            fig_c.update_layout(
                margin=dict(l=0, r=0, t=40, b=0),
                legend_title="Metric"
            )
            st.plotly_chart(fig_c, use_container_width=True)

# ---------------------------------------------
# 🩵 FOOTER
# ---------------------------------------------
st.markdown("---")
st.caption("Built with **pandas**, **Plotly**, and **Streamlit** — © 2025 Lily Patamia")
