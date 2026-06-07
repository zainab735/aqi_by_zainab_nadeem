# 🌬️ Pearls AQI Predictor

<div align="center">

**An End-to-End Machine Learning Pipeline for Air Quality Index (AQI) Forecasting**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success.svg)]()

[Live Demo 🚀](https://aqibyzainabnadeem-nqp79mn4da2ieucv2smgag.streamlit.app/) |
[Report 📄](#-final-report) |
[Dataset 📊](#-data-pipeline)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Data Pipeline](#-data-pipeline)
- [Model Training](#-model-training)
- [Dashboard](#-dashboard)
- [Model Performance](#-model-performance)
- [Key Findings](#-key-findings)
- [Future Improvements](#-future-improvements)
- [Final Report](#-final-report)
- [Author](#-author)
- [License](#-license)

---

## 🎯 Overview

The **Pearls AQI Predictor** is a complete, serverless Machine Learning Operations (MLOps) project that forecasts the Air Quality Index (AQI) for **Islamabad, Pakistan** for the next 3 days. 

This project demonstrates a production-ready ML pipeline that:
- 🔄 Automatically fetches real-time air quality data
- 🧠 Trains a Random Forest model to predict future AQI
- 📊 Provides an interactive web dashboard with forecasts
- 🧪 Uses SHAP for model explainability
- 🚨 Issues health alerts for hazardous air quality levels

**City:** Islamabad, Pakistan (33.6844° N, 73.0479° E)  
**Forecast Horizon:** 72 hours (3 days)  
**Update Frequency:** Hourly data ingestion

---

## ✨ Features

### 🔮 Predictive Capabilities
- **3-Day AQI Forecast**: Hourly predictions for the next 72 hours
- **Real-time Monitoring**: Current AQI status with live data
- **Multi-pollutant Analysis**: Tracks PM2.5, PM10, NO₂, and O₃ levels

### 🧠 Machine Learning
- **Random Forest Regressor**: Robust ensemble model with 100 trees
- **Feature Engineering**: 10 carefully engineered features including time-based and lag features
- **Model Explainability**: SHAP values to understand prediction drivers

### 📊 Interactive Dashboard
- **4 Main Sections**: Dashboard, Forecast, Analysis, and EDA
- **Visual AQI Gauge**: Color-coded gauge showing current air quality
- **Health Recommendations**: Personalized advice based on AQI levels
- **Hazard Alerts**: Automatic warnings for unhealthy air quality
- **Downloadable Forecasts**: Export predictions as CSV

### 🎨 User Experience
- **Clean White Design**: Professional, modern interface
- **Responsive Layout**: Works on desktop and mobile
- **Interactive Charts**: Zoom, pan, and hover over data points
- **Color-coded Status**: Visual indicators for air quality categories

---

## 🏗️ Architecture
──────────────────────────────────────────────────────────┐
│ DATA COLLECTION LAYER │
│ ┌──────────────────────────────────────────────────────┐ │
│ │ Open-Meteo Air Quality API (Free, No API Key) │ │
│ │ - Hourly AQI, PM2.5, PM10, NO₂, O₃ data │ │
│ │ - 90 days of historical data │ │
│ └──────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────┐
│ FEATURE ENGINEERING LAYER │
│ ┌──────────────────────────────────────────────────────┐ │
│ │ Time Features: hour, day_of_week, month │ │
│ │ Lag Features: aqi_lag_1, aqi_lag_24 │ │
│ │ Derived Features: aqi_change_rate │ │
│ │ Target: target_aqi_24h (AQI 24 hours ahead) │ │
│ └──────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────┐
│ MODEL TRAINING LAYER │
│ ┌──────────────────────────────────────────────────────┐ │
│ │ Algorithm: Random Forest Regressor │ │
│ │ Training Data: 2,160 hourly samples │ │
│ │ Train/Test Split: 80/20 │ │
│ │ Model Storage: joblib (.pkl) │ │
│ └──────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────┐
│ DEPLOYMENT LAYER │
│ ┌──────────────────────────────────────────────────────┐ │
│ │ Streamlit Web Application │ │
│ │ - Interactive Dashboard │ │
│ │ - Real-time Predictions │ │
│ │ - SHAP Explainability │ │
│ │ - Hosted on Streamlit Cloud (Free) │ │
│ └──────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘

---

## 🛠️ Technology Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Programming Language** | Python 3.9+ | Core development |
| **Data Collection** | Open-Meteo API | Free air quality data |
| **Data Processing** | Pandas, NumPy | Data manipulation |
| **Machine Learning** | Scikit-learn | Random Forest model |
| **Model Storage** | Joblib | Model serialization |
| **Explainability** | SHAP | Feature importance |
| **Visualization** | Plotly, Matplotlib | Interactive charts |
| **Web Framework** | Streamlit | Dashboard interface |
| **Hosting** | Streamlit Cloud | Free deployment |
| **Version Control** | Git, GitHub | Code management |

## Python Libraries
streamlit==1.28.0
pandas==2.1.0
numpy==1.24.0
scikit-learn==1.3.0
shap==0.43.0
plotly==5.18.0
matplotlib==3.8.0
joblib==1.3.0
requests==2.31.0

## Project Structure
AQI_by_zainab_nadeem/
│
├── 📊 Data & Models/
│   ├── aqi_data.csv              # Historical AQI data (90 days)
│   ├── aqi_model.pkl             # Trained Random Forest model
│   └── feature_names.json        # List of features used
│
├── 💻 Source Code/
│   ├── dashboard.py              # Main Streamlit application
│   ├── feature_pipeline.py       # Data collection & feature engineering
│   ├── training_pipeline.py      # Model training script
│   └── backfill.py               # Historical data backfill
│
├── 📝 Documentation/
│   ├── README.md                 # This file
│   └── REPORT.md                 # Detailed project report
│
├── ⚙️ Configuration/
│   ├── requirements.txt          # Python dependencies
│   └── .gitignore               # Git ignore file
│
└── 🚀 Deployment/
    └── .streamlit/              # Streamlit configuration
        └── config.toml
💾 Installation
-Prerequisites
-Python 3.9 or higher
-Git
-pip (Python package manager)
-Step-by-Step Setup
-1.Clone the Repository
bash
-git clone https://github.com/Zainab_735/AQI_by_zainab_nadeem.git
-cd AQI_by_zainab_nadeem
-2.Create Virtual Environment (Recommended)
bash
# Windows
-python -m venv venv
-venv\Scripts\activate

# macOS/Linux
-python3 -m venv venv
-source venv/bin/activate
-3.Install Dependencies
-pip install -r requirements.txt
-4.Verify Installation
-python -c "import streamlit, sklearn, shap, plotly; print('✅ All packages installed!')"
 ## Usage
-Running the Dashboard Locally
bash
- streamlit run dashboard.py
-The application will open in your browser at http://localhost:8501

# Training a New Model
If you want to retrain the model with fresh data:
Step 1: Fetch latest data and engineer features
python feature_pipeline.py

Step 2: Train the model
python training_pipeline.py
Customizing for Your City
Edit the coordinates in feature_pipeline.py:
CITY_NAME = "Your City" # Mine was Islamabad
CITY_LAT = 33.6844  # Your latitude
CITY_LON = 73.0479  # Your longitude

📊 Data Pipeline
Data Source
Open-Meteo Air Quality API
🌐 Website: https://open-meteo.com/
💰 Cost: Completely free, no API key required
📈 Coverage: Global, hourly resolution
🔄 Updates: Every hour
Data Collection Process
Historical Data Fetch
Retrieves 90 days of hourly air quality data
Total: 2,160 data points (90 days × 24 hours)
Variables: US AQI, PM2.5, PM10, NO₂, O₃
Feature Engineering
<img width="904" height="571" alt="image" src="https://github.com/user-attachments/assets/3dcc0a91-437a-4b0a-ad67-2bbec06b8a2c" />
Target Variable
target_aqi_24h: AQI value 24 hours in the future
Used for training the predictive model
🧠 Model Training
Algorithm: Random Forest Regressor
Why Random Forest?
✅ Handles non-linear relationships well
✅ Robust to outliers
✅ No feature scaling required
✅ Provides feature importance
✅ Works well with mixed feature types
Hyperparameters
RandomForestRegressor(
    n_estimators=100,      # Number of trees
    random_state=42,       # Reproducibility
    n_jobs=-1              # Use all CPU cores
)
