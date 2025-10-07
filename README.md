# Environmental_Health_Index

**data as empathy.** a visual + ml exploration of how environmental stress (co₂, pollution, deforestation) relates to human health (life expectancy, chronic disease).

## quick start
```bash
# 1) install deps
pip install -r requirements.txt

# 2) add data (see below)
mkdir -p data output

# 3) build the dataset
python scripts/build_dataset.py

# 4) run dashboard
streamlit run app.py