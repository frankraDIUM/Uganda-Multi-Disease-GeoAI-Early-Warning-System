# 🦠 Uganda Multi-Disease GeoAI Early Warning System
A Climate-Intelligent Early Warning System for Vector-Borne and Waterborne Diseases in Uganda

**Summary**

This project is a complete GeoAI-powered early warning system that predicts vector-borne (e.g., malaria) 
and waterborne (e.g., cholera) disease risk at the district level in Uganda. 
It integrates high-resolution climate data from Copernicus ERA5-Land, administrative boundaries, engineered spatiotemporal features, 
and machine learning models to generate risk predictions and outbreak alerts. 
The system is delivered through an interactive Streamlit dashboard with real-time forecasting and mapping capabilities.

**Key Objectives Achieved**

- Fuse climate (ERA5-Land) and spatial data (OSM + HDX boundaries)
- Build separate predictive models for vector-borne and waterborne diseases
- Develop an epidemiological outbreak detection engine
- Create a production-grade interactive dashboard with AI recommendations
- Simulate realistic early warning workflows (risk scoring, alerts, mapping)

**Technical Architecture**

Data Sources:

- Climate: ERA5-Land Monthly Means (temperature, precipitation)
- Spatial: Uganda ADM2 districts (135 districts)
- Health: Advanced synthetic epidemiological simulation (population-adjusted, spatial spillover, rolling outbreak detection)

Core Technologies:

Python, GeoPandas, xarray, rioxarray

Scikit-learn (RandomForestRegressor + RandomForestClassifier)

Streamlit + Plotly + Folium

Joblib model persistence

Modeling Performance (Test Set)

Vector-Borne Model: R² = 0.861, MAE = 9.12
Waterborne Model: R² = 0.738, MAE = 5.06
Outbreak Classifier: AUC = 0.936

**Key Features Engineered**

- Temporal lags, rolling statistics, anomalies
- Climate-disease interaction terms
- Seasonal and long-term trend components
- Spatial spillover simulation
- Incidence rates (per 100,000 population)

**Dashboard Capabilities**

- District-level interactive exploration
- Real-time AI predictions using trained models
- Risk mapping (Folium + Plotly)
- Outbreak probability & early warning alerts
- Data export functionality
