# Forecasting Federal Reserve Policy Decisions Using Machine Learning

## Category
Statistical Analysis Tools

## Problem Statement
The Federal Reserve’s (Fed) monetary policy decisions strongly influence the economy and financial markets. While the Taylor rule proposes a simple relationship between the policy rate, inflation, and the output gap, it cannot fully capture the complexity and discretion involved in actual policy decisions.

This project aims to predict the **level of the Fed’s policy rate** (rate level prediction) at each FOMC meeting using **multi-feature machine learning models**. Unlike classification approaches that only predict the direction of rate changes, this regression-based approach allows for forecasting the exact policy rate and enables a direct comparison between machine learning models and the Taylor rule as a baseline.

## Motivation
The main motivation of this project is to explore whether machine learning methods can better replicate the Federal Reserve’s policy behavior than traditional rule-based approaches. By leveraging economic and financial data, this project combines programming and data analysis skills to build interpretable and data-driven models, while assessing whether modern ML techniques provide meaningful improvements over classical econometric benchmarks.

## Planned Approach and Technologies
The project follows a structured machine learning workflow:

- **Data Collection and Preprocessing**: Macroeconomic and financial time series from FRED, including inflation, unemployment, industrial production (as an output gap proxy), yield curve measures, and financial spreads, along with historical FOMC policy rates. All features are aligned with FOMC meeting dates using only information available prior to each meeting. Lagged variables, rolling averages, and normalization are applied.

- **Models**: Regularized linear models (Ridge and Lasso regression) as well as non-linear ensemble methods (Random Forest and XGBoost).

- **Baseline**: A Taylor rule estimated via OLS regression, used solely as a benchmark.

- **Evaluation**: Temporal validation with a training period from 2000–2015 and a test period from 2016–2025. Model performance is evaluated using RMSE, MAE, and R². Model interpretability is assessed through feature importance and SHAP-style analyses where applicable.

## Expected Challenges and Mitigation
- **Small sample size (~200 observations)**: Addressed through regularization, limited model complexity, and strict time-based validation.
- **Differences in data frequency and publication lags**: Handled by harmonizing series using the most recent available values or rolling averages.
- **Non-stationarity and regime shifts**: Mitigated by including financial indicators and testing robustness across subsamples.
- **Interpretability of complex models**: Addressed through feature importance and post-estimation analysis.

## Success Criteria
The project is successful if machine learning models can accurately predict the Fed’s policy rate out-of-sample and demonstrate improved performance relative to the Taylor rule baseline. Estimated relationships should remain economically plausible, and the project should result in clean, reproducible, and well-documented code, supported by clear diagnostics and visualizations.

## Stretch Goals
- Compare different prediction horizons (next meeting versus multiple meetings ahead).
- Identify the most influential variables driving policy decisions using interpretability tools.
- Extend the analysis to other central banks (e.g., the Swiss National Bank).
- Explore the use of LSTM models to capture deeper temporal dynamics in macroeconomic data.



