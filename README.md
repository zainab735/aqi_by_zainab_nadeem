# aqi_by_zainab_nadeem
# Pearls AQI Predictor - Final Report

## Executive Summary
A serverless ML pipeline that forecasts AQI for Islamabad for the next 3 days using Random Forest and SHAP explainability.

## Technology Stack
- **Data Source**: Open-Meteo Air Quality API (Free, no API key)
- **ML Framework**: Scikit-learn (Random Forest)
- **Explainability**: SHAP
- **Dashboard**: Streamlit
- **Hosting**: Streamlit Cloud (Free)

## Data Pipeline
- Fetches 90 days of historical AQI data
- Engineers time-based features (hour, day, month)
- Creates lag features (1h, 24h) and change rate
- Total: 2,160 hourly data points

## Model Performance
- **RMSE**: [Copy from Colab output]
- **MAE**: [Copy from Colab output]
- **R²**: [Copy from Colab output]

## Key Findings
1. PM2.5 is the strongest predictor of AQI
2. AQI peaks during morning and evening rush hours
3. Weekend AQI is typically lower than weekdays

## Dashboard Features
- Real-time AQI with hazard alerts
- 3-day forecast visualization
- SHAP explainability plots
- Interactive EDA charts

## Live Dashboard
[Insert your Streamlit Cloud URL here]
