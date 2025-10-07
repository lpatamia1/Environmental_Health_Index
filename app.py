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
# 📊 QUICK DATA SUMMARY
# ---------------------------------------------
st.info(f"""
**Dataset Coverage:** {df['country'].nunique()} countries  
**Years:** {df['year'].min()}–{df['year'].max()}  
**Total Records:** {len(df):,}  
**Metrics:** CO₂, PM2.5, Forest Loss, Life Expectancy, Chronic Disease, Eco-Health Score
""")

# ---------------------------------------------
# 🧭 TABS LAYOUT
# ---------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Global Overview",
    "Global Trends",
    "Correlations",
    "Machine Learning Insights"
])

# ensure output folder for visuals
os.makedirs("output/charts", exist_ok=True)

# ---------------------------------------------
# 🧭 HEADER
# ---------------------------------------------
st.title("🌍 Global Eco-Health Dashboard")
st.caption("Data as empathy — revealing how our planet’s well-being reflects our own.")
st.markdown(
    """
    This interactive dashboard explores the intricate connections between environmental health
    indicators and human well-being across countries and over time. Using a curated dataset
    combining CO₂ emissions, air quality (PM2.5), forest loss, and health metrics like life expectancy
    and chronic disease rates, we visualize global trends and correlations.

    **Use the sidebar controls** to select the year, choose metrics for the choropleth map,
    and highlight specific countries for detailed trend analysis.

    *Data sources include Our World in Data, WHO, and World Bank.*
    """
)
st.markdown("---")

# ---------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------
st.sidebar.header("Controls")

years = sorted(df["year"].dropna().unique())
min_year, max_year = int(min(years)), int(max(years))
year = st.sidebar.slider("Select Year", min_year, max_year, max_year, step=1)

metric_opts = {
    "Eco-Health Score": "eco_health_score",
    "Life Expectancy": "life_expectancy",
    "PM2.5 (μg/m³)": "pm25",
    "CO₂ per Capita (tons)": "co2_per_capita",
    "Forest Loss (%)": "forest_loss_pct",
    "Chronic Disease Rate": "chronic_disease_rate",
}
map_metric_label = st.sidebar.selectbox("Choropleth Metric", list(metric_opts.keys()), index=0)
map_metric = metric_opts[map_metric_label]

focus_countries = st.sidebar.multiselect(
    "Highlight Countries",
    sorted(df["country"].unique())[:300],
)

# ---------------------------------------------
# GLOBAL MAP
# ---------------------------------------------
st.subheader(f"Global Map — {map_metric_label} ({year})")

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
        title=None,
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar_title=map_metric_label,
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # Top and bottom 5 analysis
    top5 = df_y.nlargest(5, map_metric)[["country", map_metric]]
    bottom5 = df_y.nsmallest(5, map_metric)[["country", map_metric]]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**⬆️ Top 5 Countries**")
        st.table(top5)
    with col2:
        st.markdown("**⬇️ Bottom 5 Countries**")
        st.table(bottom5)

    st.markdown("---")

# 💾 Save visualization
fig_map.write_image(f"output/charts/global_map_{year}.png")

# ✍️ Analysis
st.markdown(
    f"""
    **Analysis:**  
    The global map for **{map_metric_label.lower()}** in {year} highlights disparities across regions.  
    - Countries with lighter shades show *lower* values, while darker blues indicate *higher impact*.  
    - This spatial view helps pinpoint where environmental stress or health inequality is most visible.
    """
)
st.markdown("---")

# ---------------------------------------------
# CO₂ VS LIFE EXPECTANCY (GLOBAL TREND)
# ---------------------------------------------
st.subheader("CO₂ vs Life Expectancy — Global Trend")
with tab2:
    st.subheader("📈 CO₂ vs Life Expectancy — Global Trend")
    # (your dual-axis plot + analysis)
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
fig_dual.write_image("output/charts/co2_vs_life_expectancy.png")

st.markdown(
    """
    **Analysis:**  
    As global CO₂ emissions per capita have risen over the decades, life expectancy has also increased —  
    but this paradox reflects unequal progress. Industrialized nations experience longer lives **despite**  
    higher emissions, revealing the imbalance between environmental cost and human development.
    """
)
st.markdown("---")

# ---------------------------------------------
# CORRELATION HEATMAP
# ---------------------------------------------
st.subheader(f"Correlations Between Environment & Health ({year})")
with tab3:
    st.subheader(f"🔥 Correlations Between Environment & Health ({year})")
    # (your correlation heatmap + analysis)

corr_cols = ["co2_per_capita", "pm25", "forest_loss_pct",
             "life_expectancy", "chronic_disease_rate", "eco_health_score"]
corr_df = df_y[corr_cols].copy()
corr = corr_df.corr()

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
fig_corr.write_image(f"output/charts/correlation_heatmap_{year}.png")

st.markdown(
    """
    **Analysis:**  
    - Strong *negative* correlations indicate that higher pollution and CO₂ emissions reduce life expectancy.  
    - Positive links with the **eco-health score** confirm that better environmental quality aligns  
      with longer, healthier lives.
    """
)
st.markdown("---")

# ---------------------------------------------
# FEATURE IMPORTANCE (ML MODEL)
# ---------------------------------------------
st.subheader("Key Drivers of Life Expectancy")
st.caption("Machine learning insights from Random Forest Regression")
with tab4:
    st.subheader("🧠 Key Drivers of Life Expectancy")
    # (your Random Forest + bar chart)

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
        st.metric("Model R² (In-Sample)", f"{r2:.2f}")
        st.caption("Diagnostic only — for exploration.")
    with col2:
        fig_imp = px.bar(
            importances,
            orientation="h",
            title=None,
            color=importances,
            color_continuous_scale="Blues_r",
        )
        fig_imp.update_layout(
            xaxis_title="Importance",
            yaxis_title=None,
            coloraxis_showscale=False,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color="#1E1E1E", size=14),
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_imp, use_container_width=True)
        fig_imp.write_image("output/charts/feature_importance.png")
    st.markdown(
        """
        **Analysis:**  
        The Random Forest model finds that **eco-health score** and **air quality (PM2.5)**  
        are the strongest predictors of life expectancy — reinforcing how cleaner, balanced ecosystems  
        sustain longer and healthier human lives.
        """
    )
else:
    st.info("Not enough data to train model yet — add more datasets in /data and rebuild.")


# ---------------------------------------------
# COUNTRY DETAIL VIEW
# ---------------------------------------------
if focus_countries:
    st.subheader("Country Detail Trends")
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
        fig_c.write_image(f"output/charts/{c}_trend.png")



# ---------------------------------------------
# 🩵 FOOTER
# ---------------------------------------------

st.caption("Built with **pandas**, **Plotly**, and **Streamlit** — © 2025 Lily Patamia")
