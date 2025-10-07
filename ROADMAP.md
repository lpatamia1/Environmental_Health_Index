## Phase 1 — Data Integration ✅

Focus: Build a unified dataset that captures environmental and health variables across countries and years.
Key steps:

Collect CO₂, PM2.5 (air quality), and life expectancy datasets from open sources.

Clean, merge, and standardize data by ISO country codes and years.

Create derived indicators like the Eco-Health Score (a composite measure of environmental well-being).

Validate data coverage and consistency.

## Phase 2 — Dashboard Development ✅

Focus: Design a user-friendly, visually compelling Streamlit app for global exploration.
Key steps:

Build interactive choropleth maps and trend visualizations with Plotly.

Add correlation heatmaps and dual-axis plots to connect environmental and health trends.

Implement tabs and sidebar controls for smooth user navigation.

Export charts automatically to the /output/charts folder for analysis and reporting.

## Phase 3 — Machine Learning Insights ✅

Focus: Explore which environmental features most strongly predict human health.
Key steps:

Train a Random Forest Regressor on life expectancy using CO₂, PM2.5, and eco-health metrics.

Evaluate feature importance to reveal main drivers of health outcomes.

Interpret model results to balance technical accuracy with empathy-driven storytelling.

## Phase 4 — Refinement & Expansion 🚧

Focus: Extend the dashboard’s scope and improve accessibility.
Next goals:

Add new variables (e.g., access to clean water, waste management, or biodiversity).

Incorporate socioeconomic context (income, population density).

Optimize UI/UX for deployment on Streamlit Cloud or Hugging Face Spaces.

Write documentation for open data contributors and reproducibility.