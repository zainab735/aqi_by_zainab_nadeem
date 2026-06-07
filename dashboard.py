import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import os

st.set_page_config(page_title="Pearls AQI Predictor", layout="wide", page_icon="🌬️")

# Load model and data
@st.cache_resource
def load_model():
    return joblib.load('aqi_model.pkl')

@st.cache_data
def load_data():
    df = pd.read_csv('aqi_data.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

@st.cache_data
def load_features():
    with open('feature_names.json', 'r') as f:
        return json.load(f)

model = load_model()
df = load_data()
features_to_use = load_features()

# Header
st.title("🌬️ Pearls AQI Predictor")
st.subheader("3-Day Air Quality Forecast for Islamabad")

# Current AQI
latest_data = df.sort_values('timestamp').tail(1)
current_aqi = latest_data['us_aqi'].iloc[0]

# Alert system
if current_aqi > 150:
    st.error(f"🚨 HAZARDOUS AIR QUALITY ALERT! Current AQI: {int(current_aqi)}")
    st.write("Limit outdoor activities. Wear N95 masks if going outside.")
elif current_aqi > 100:
    st.warning(f"⚠️ Unhealthy for Sensitive Groups. Current AQI: {int(current_aqi)}")
    st.write("Children, elderly, and people with respiratory issues should limit outdoor exposure.")
elif current_aqi > 50:
    st.info(f"ℹ️ Moderate Air Quality. Current AQI: {int(current_aqi)}")
else:
    st.success(f"✅ Good Air Quality! Current AQI: {int(current_aqi)}")

# 3-Day Forecast
st.markdown("---")
st.subheader("📈 3-Day AQI Forecast")

forecast_df = latest_data.copy()
predictions_list = []

for i in range(72):  # 72 hours = 3 days
    next_hour = forecast_df['timestamp'].iloc[-1] + timedelta(hours=1)
    new_row = forecast_df.iloc[-1].copy()
    new_row['timestamp'] = next_hour
    new_row['hour'] = next_hour.hour
    new_row['day_of_week'] = next_hour.dayofweek
    new_row['month'] = next_hour.month
    new_row['aqi_lag_1'] = new_row['us_aqi']
    
    X_pred = new_row[features_to_use].values.reshape(1, -1)
    pred_aqi = max(0, model.predict(X_pred)[0])
    
    new_row['us_aqi'] = pred_aqi
    new_row['aqi_change_rate'] = pred_aqi - forecast_df['us_aqi'].iloc[-1]
    
    forecast_df = pd.concat([forecast_df, pd.DataFrame([new_row])], ignore_index=True)
    
    if (i + 1) % 24 == 0:
        predictions_list.append({
            'Day': f"Day {(i+1)//24}",
            'Date': next_hour.strftime('%Y-%m-%d'),
            'Predicted AQI': int(pred_aqi)
        })

# Plot forecast
fig, ax = plt.subplots(figsize=(12, 5))
historical = df.tail(48)
ax.plot(historical['timestamp'], historical['us_aqi'], label='Historical (48h)', color='blue', linewidth=2)
ax.plot(forecast_df['timestamp'], forecast_df['us_aqi'], label='Forecast (72h)', color='red', linewidth=2, linestyle='--')
ax.axvline(x=historical['timestamp'].iloc[-1], color='green', linestyle='-', linewidth=2, label='Now')
ax.set_ylabel('AQI', fontsize=12)
ax.set_xlabel('Time', fontsize=12)
ax.set_title('Historical and 3-Day AQI Forecast', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

# Daily summary
st.markdown("### 📅 Daily Forecast Summary")
pred_df = pd.DataFrame(predictions_list)
st.dataframe(pred_df, use_container_width=True)

# SHAP Explainability
st.markdown("---")
st.subheader("🧠 Model Explainability (SHAP)")
st.write("Understanding what factors influence the AQI prediction:")

with st.spinner("Calculating feature importance..."):
    X_sample = df[features_to_use].sample(50, random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    fig_shap, ax_shap = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, feature_names=features_to_use, show=False)
    st.pyplot(fig_shap)

# Feature importance bar chart
st.markdown("### 📊 Feature Importance")
feature_importance = pd.DataFrame({
    'Feature': features_to_use,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

fig_imp, ax_imp = plt.subplots(figsize=(10, 5))
ax_imp.barh(feature_importance['Feature'], feature_importance['Importance'], color='skyblue')
ax_imp.set_xlabel('Importance', fontsize=12)
ax_imp.set_title('Feature Importance in AQI Prediction', fontsize=14, fontweight='bold')
ax_imp.invert_yaxis()
plt.tight_layout()
st.pyplot(fig_imp)

# EDA Section
st.markdown("---")
st.subheader("📊 Exploratory Data Analysis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### AQI by Hour of Day")
    hourly_avg = df.groupby('hour')['us_aqi'].mean()
    fig_hour, ax_hour = plt.subplots(figsize=(8, 4))
    ax_hour.bar(hourly_avg.index, hourly_avg.values, color='coral')
    ax_hour.set_xlabel('Hour')
    ax_hour.set_ylabel('Average AQI')
    ax_hour.set_title('Daily AQI Pattern')
    plt.tight_layout()
    st.pyplot(fig_hour)

with col2:
    st.markdown("#### AQI by Day of Week")
    df['day_name'] = df['timestamp'].dt.day_name()
    daily_avg = df.groupby('day_name')['us_aqi'].mean()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_avg = daily_avg.reindex(day_order)
    fig_day, ax_day = plt.subplots(figsize=(8, 4))
    ax_day.bar(daily_avg.index, daily_avg.values, color='lightgreen')
    ax_day.set_xlabel('Day')
    ax_day.set_ylabel('Average AQI')
    ax_day.set_title('Weekly AQI Pattern')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig_day)

# Pollutant correlation
st.markdown("#### Pollutant Correlations")
corr_cols = ['us_aqi', 'pm10', 'pm2_5', 'nitrogen_dioxide', 'ozone']
corr_matrix = df[corr_cols].corr()
fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
im = ax_corr.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
ax_corr.set_xticks(range(len(corr_cols)))
ax_corr.set_yticks(range(len(corr_cols)))
ax_corr.set_xticklabels(corr_cols, rotation=45, ha='right')
ax_corr.set_yticklabels(corr_cols)
for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        ax_corr.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', ha='center', va='center', color='black')
plt.colorbar(im)
plt.title('Pollutant Correlation Matrix')
plt.tight_layout()
st.pyplot(fig_corr)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
<p>🌍 Pearls AQI Predictor | Built with Streamlit, Scikit-learn, and SHAP</p>
<p>Data Source: Open-Meteo Air Quality API | Model: Random Forest Regressor</p>
</div>
""")