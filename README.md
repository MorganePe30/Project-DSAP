# Fed Policy ML Project  
**Forecasting Federal Reserve Policy Decisions Using Machine Learning**

## Research Question
Can machine learning models outperform a standard Taylor-rule-style benchmark in predicting the Federal Reserve’s target policy rate at FOMC meetings?

This project compares a traditional economic baseline with modern machine learning models to evaluate predictive performance, robustness, and interpretability.

---

## Project Overview
This project builds a fully reproducible machine learning pipeline to forecast the Federal Funds target rate using U.S. macroeconomic data available **prior** to FOMC meetings.

The main objectives are:
- Compare an economic benchmark (Taylor rule) with ML models
- Prevent data leakage through careful time alignment
- Ensure reproducibility and clean project structure
- Provide a single entry point via `python main.py`

---

## Data
All data are sourced from **FRED (Federal Reserve Economic Data)**.

### Target Variable
- Federal Funds target rate:
  - **DFEDTAR**: single target rate (pre-2008)
  - **DFEDTARU**: upper bound of the target range (post-2008)

These two series are concatenated to create a consistent policy target spanning multiple monetary policy regimes.

### Macroeconomic Features (monthly, lagged)
- Effective Federal Funds Rate
- CPI (inflation)
- Core PCE
- Unemployment rate
- Industrial production
- Yield curve slope (10y – 3m Treasury)
- Corporate credit spread (Baa – 10y Treasury)

All features are constructed so that **only information available before each FOMC meeting is used**, avoiding look-ahead bias.

---

## Models
The following models are implemented and compared:
- **Taylor Rule (OLS)** — economic baseline
- **Ridge Regression**
- **Lasso Regression**
- **Random Forest**
- **XGBoost**

Models are evaluated using a time-based split:
- **Training period:** 2000–2015  
- **Test period:** 2016–2025  

Evaluation metrics include RMSE, MAE, and R².

## Results Summary

Out-of-sample results (2016–2025) show that the Taylor-rule-style OLS benchmark performs competitively in predicting the Federal Reserve’s policy rate level. Linear regularized models (Ridge and Lasso) deliver comparable performance, while tree-based models (Random Forest and XGBoost) tend to overfit the limited sample and exhibit weaker generalization.

These findings highlight that, in a small-sample and highly structured macroeconomic setting, simple and economically grounded models can rival or outperform more complex machine learning approaches. This result underscores the importance of model parsimony, proper regularization, and careful temporal validation in applied monetary policy forecasting.


---

## Project Structure
Project-DSAP/
├── main.py # Main entry point (must run)
├── README.md # Project documentation
├── PROPOSAL.md # Approved project proposal
├── environment.yml # Conda environment (reproducibility)
├── data/
│ ├── raw/ # Raw FRED CSV files
│ └── processed/ # ML-ready dataset (generated)
├── src/
│ ├── data/ # Data loading
│ ├── features/ # Feature engineering
│ ├── models/ # Model training scripts
│ └── evaluation/ # Plots, tables, diagnostics
├── results/
│ ├── model_comparison.csv
│ ├── ensemble_results.csv
│ ├── true_vs_predicted.png
│ └── time_series_predictions.png
├── notebooks/ # Exploratory analysis
└── tests/


---

## Environment Setup (Conda)
This project uses **Conda**, as required by the course.

To recreate the environment locally:
```bash
conda env create -f environment.yml
conda activate fed-ml-env
```


## Running the Project

Once the Conda environment is activated, run:
```bash
python main.py
```



## Reproducibility

All scripts use fixed random seeds where applicable.  
Running `python main.py` reproduces the full pipeline: data loading, feature construction, model training, evaluation, and figure generation.




