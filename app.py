# UGANDA MULTI-DISEASE EARLY WARNING SYSTEM

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import joblib
import os
import json
import requests

# SIMULATED ALERT SYSTEM
def send_simulated_alert(message, to_phone):
    if "sms_log" not in st.session_state:
        st.session_state.sms_log = []

    st.session_state.sms_log.append({
        "time": pd.Timestamp.now(),
        "to": to_phone,
        "message": message,
        "status": "SIMULATED_SENT"
    })

# ALERT HISTORY STORAGE
if "alert_history" not in st.session_state:
    st.session_state.alert_history = []

if "sms_sent_districts" not in st.session_state:
    st.session_state.sms_sent_districts = set()

@st.dialog("NATIONAL DISEASE SURVEILLANCE SYSTEM")
def confirm_sms_send(message, district, level):

    st.markdown("""
    <style>
    .alert-box {
        background: linear-gradient(135deg, #2b0000, #5c0000);
        padding: 20px;
        border-radius: 15px;
        border: 2px solid #ff4d4d;
        color: white;
        animation: pulseBorder 1.5s infinite;
    }
    @keyframes pulseBorder {
        0% { box-shadow: 0 0 5px rgba(255,0,0,0.4); }
        50% { box-shadow: 0 0 20px rgba(255,0,0,0.9); }
        100% { box-shadow: 0 0 5px rgba(255,0,0,0.4); }
    }
    .alert-title {
        font-size: 15px;
        font-weight: bold;
        margin-bottom: 8px;
    }
    .alert-sub {
        font-size: 14px;
        opacity: 0.85;
    }
    .alert-message {
        margin-top: 15px;
        padding: 10px;
        background: rgba(0,0,0,0.3);
        border-left: 4px solid #ff4d4d;
        font-size: 15px;
    }
    .siren {
        font-size: 26px;
        animation: blink 1s infinite;
    }
    @keyframes blink {
        50% { opacity: 0.3; }
    }
    </style>
    """, unsafe_allow_html=True)

    st.html(f"""
    <div class="alert-box">
        <div class="alert-title">
            <span class="siren">🚨</span> {level} OUTBREAK ALERT
        </div>
        <div class="alert-sub">
            Ministry of Health • Real-Time Surveillance Engine
        </div>
        <hr style="border:0.5px solid rgba(255,255,255,0.2);">
        <b> District:</b> {district} <br>
        <b> Threat Level:</b> {level} <br>
        <b> Detection Source:</b> AI Climate-Disease Model
        <div class="alert-message">
            {message}
        </div>
    </div>
    """)

    st.markdown("""
    <style>
    div.stButton > button {
        font-size: 13px !important;
        padding: 0.4rem 0.8rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("### 📲 Emergency Broadcast Authorization")
    st.write("You are about to send an outbreak warning to local health authorities and residents.")
    st.caption("Close this dialog without authorizing to cancel.")

    # AUTHORIZE
    if st.button("AUTHORIZE EMERGENCY SMS", use_container_width=True, type="primary"):
        send_simulated_alert(
            message=message,
            to_phone=f"District Health Officer - {district}"
        )
        st.session_state.sms_sent_districts.add(district)
        st.session_state.toast_msg = f"📩 SMS sent to {district}"
        st.session_state.pending_sms_district = None

        # Register in alert history
        st.session_state.alert_history.append({
            "time": pd.Timestamp.now(),
            "district": district,
            "risk_index": st.session_state.get("pending_sms_risk", "N/A"),
            "message": message,
            "status": "📩 SMS SENT"
        })

        st.rerun()

# INTERFACE
st.set_page_config(
    page_title="Uganda Disease Risk Dashboard",
    layout="wide",
    page_icon="🦠"
)

st.markdown(
    "<h2 style='margin-bottom: 0;'>🦠 Uganda Multi-Disease Early Warning System</h2>",
    unsafe_allow_html=True
)

st.markdown("**Climate-Driven Prediction of Vector-Borne & Waterborne Diseases**")

# LOAD DATA
@st.cache_data
def load_data():
    path = r"I:\Data Science Projects\disease_prediction\data\processed\final_modeling_dataset_READY.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df

# PHASE 6 FEATURES
def create_model_features(df):
    df = df.sort_values(["district", "date"]).copy()

    df["temp_precip_interaction"] = df["t2m"] * df["tp"]

    df["tp_roll3"] = df.groupby("district")["tp"].transform(
        lambda x: x.rolling(3,1).mean()
    )

    df["t2m_roll3"] = df.groupby("district")["t2m"].transform(
        lambda x: x.rolling(3,1).mean()
    )

    df["tp_anomaly_lag1"] = df.groupby("district")["tp_anomaly"].shift(1)
    df["t2m_anomaly_lag1"] = df.groupby("district")["t2m_anomaly"].shift(1)

    df[["tp_anomaly_lag1","t2m_anomaly_lag1"]] = df.groupby("district")[[
        "tp_anomaly_lag1","t2m_anomaly_lag1"
    ]].transform(lambda x: x.bfill().ffill())

    return df

@st.cache_data
def get_final_dataset():
    df = load_data()
    df = create_model_features(df)

    geo_path = r"I:\Data Science Projects\disease_prediction\data\raw\boundaries\uganda_districts.geojson"
    if "latitude" not in df.columns or "longitude" not in df.columns:
        with open(geo_path, "r", encoding="utf-8") as f:
            geo = json.load(f)

        centroid_map = {
            feature["properties"]["adm2_name"]: (
                feature["properties"]["center_lat"],
                feature["properties"]["center_lon"]
            )
            for feature in geo["features"]
            if feature["properties"].get("adm2_name") is not None
            and feature["properties"].get("center_lat") is not None
            and feature["properties"].get("center_lon") is not None
        }

        df["latitude"] = df["district"].map(
            lambda x: centroid_map.get(x, (np.nan, np.nan))[0]
        )

        df["longitude"] = df["district"].map(
            lambda x: centroid_map.get(x, (np.nan, np.nan))[1]
        )

    return df.dropna().reset_index(drop=True)

df = get_final_dataset()

geo_path = r"I:\Data Science Projects\disease_prediction\data\raw\boundaries\uganda_districts.geojson"

@st.cache_resource
def load_models():
    path = r"I:\Data Science Projects\disease_prediction\data\processed"

    rf_v = joblib.load(os.path.join(path, "model_vector_borne.pkl"))
    rf_w = joblib.load(os.path.join(path, "model_waterborne.pkl"))
    clf = joblib.load(os.path.join(path, "model_outbreak_clf.pkl"))

    return rf_v, rf_w, clf

rf_vector, rf_water, clf_outbreak = load_models()

# SAFETY CLEANING
df = df.dropna(subset=[
    "district",
    "date",
    "vector_borne_cases",
    "waterborne_cases"
])

district_groups = df.groupby("district")

# SIDEBAR FILTERS
if "pending_district" not in st.session_state:
    st.session_state.pending_district = None

st.sidebar.header("🔍 Filters")

districts = sorted(df["district"].unique())

if st.session_state.pending_district is not None:
    st.session_state["district_selector"] = st.session_state.pending_district
    st.session_state.pending_district = None

selected_district = st.sidebar.selectbox(
    "Select District",
    districts,
    key="district_selector"
)

min_date = df["date"].min().date()
max_date = df["date"].max().date()

date_range = st.sidebar.date_input(
    "Date Range",
    value=[min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

@st.cache_data
def get_prediction_data(district_name, start_date, end_date):
    if district_name in district_groups.groups:
        district_base = district_groups.get_group(district_name)
    else:
        district_base = df[df["district"] == district_name]

    mask = (
        (district_base["date"].dt.date >= start_date) &
        (district_base["date"].dt.date <= end_date)
    )
    sub = district_base[mask].sort_values("date").copy()

    sub["risk_score"] = (
        0.6 * sub["vector_borne_cases"] +
        0.4 * sub["waterborne_cases"]
    )
    sub["risk_roll7"] = sub["risk_score"].rolling(7, min_periods=1).mean()

    return sub

district_df = get_prediction_data(selected_district, date_range[0], date_range[1])

if district_df.empty:
    st.warning("No data available for this selection.")
    st.stop()

latest = district_df.iloc[-1]

# MODEL FEATURES
model_features = [
    "t2m","tp","t2m_anomaly","tp_anomaly",
    "tp_lag1","tp_lag2","temp_lag1",
    "tp_roll3","t2m_roll3",
    "tp_anomaly_lag1","t2m_anomaly_lag1",
    "temp_precip_interaction",
    "year_trend","sin_month","cos_month",
    "spatial_lag_cases"
]

missing = [f for f in model_features if f not in latest.index]

if missing:
    st.error(f"Missing features: {missing}")
    st.stop()

latest_features = latest[model_features].values.reshape(1, -1)

@st.cache_data
def run_prediction(features):
    pv = rf_vector.predict(features)[0]
    pw = rf_water.predict(features)[0]
    po = clf_outbreak.predict_proba(features)[0][1]
    return pv, pw, po

pred_vector, pred_water, pred_outbreak_prob = run_prediction(latest_features)

# UNIFIED RISK CLASSIFICATION
def classify_risk(prob):
    risk = prob * 100
    if risk > 5:
        return "🔴 CRITICAL", "CRITICAL"
    elif risk > 2:
        return "🟠 HIGH", "HIGH"
    elif risk > 1:
        return "🟡 MODERATE", "MODERATE"
    else:
        return "🟢 LOW", "LOW"

# UNCERTAINTY ESTIMATION
vector_std = np.std([
    rf_vector.predict(latest_features)[0] *
    np.random.normal(1, 0.1)
    for _ in range(30)
])

water_std = np.std([
    rf_water.predict(latest_features)[0] *
    np.random.normal(1, 0.1)
    for _ in range(30)
])

outbreak_std = np.std([
    clf_outbreak.predict_proba(latest_features)[0][1] *
    np.random.normal(1, 0.05)
    for _ in range(30)
])

metrics_container = st.container()
with metrics_container:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Surface Temp", f"{latest['t2m']:.1f}°C")
        st.caption(f"±{int(vector_std)} vector uncertainty")

    with col2:
        st.metric("Cumulative Rainfall", f"{latest['tp']:.1f} mm")
        st.caption(f"±{int(water_std)} water uncertainty")

    with col3:
        st.metric(
            "Vector Cases",
            int(latest["vector_borne_cases"]),
            delta="High"
            if latest["vector_borne_cases"] >
            district_df["vector_borne_cases"].mean()
            else "Normal"
        )
        st.caption(f"±{outbreak_std:.3f} outbreak variability")

    with col4:
        risk_label, risk_level = classify_risk(pred_outbreak_prob)
        st.metric("Outbreak Risk", risk_label)

# TOAST
if "toast_msg" in st.session_state:
    st.toast(st.session_state.toast_msg, icon="📩")
    del st.session_state.toast_msg

# ALERT LOGIC
risk_index = pred_outbreak_prob * 100
risk_label, level = classify_risk(pred_outbreak_prob)

alert_msg = None

if level == "CRITICAL":
    alert_msg = (
        f"🚨 CRITICAL: High outbreak risk in "
        f"{selected_district} ({risk_index:.2f}%)"
    )
elif level in ["HIGH", "MODERATE"]:
    alert_msg = (
        f"⚠️ WARNING: Elevated disease risk in "
        f"{selected_district} ({risk_index:.2f}%)"
    )

if "authorized_sms_districts" not in st.session_state:
    st.session_state.authorized_sms_districts = set()

if alert_msg:

    skip_alerts = st.session_state.get("suppress_alert", False)

    if not skip_alerts:
        if (
            not st.session_state.alert_history or
            st.session_state.alert_history[-1]["message"] != alert_msg
        ):
            st.session_state.alert_history.append({
                "time": pd.Timestamp.now(),
                "district": selected_district,
                "risk_index": round(risk_index, 2),
                "message": alert_msg,
                "status": "⚠️ DETECTED"
            })

    now = pd.Timestamp.now()

    if "alert_time" not in st.session_state:
        st.session_state.alert_time = now

    if "last_alert_msg" not in st.session_state:
        st.session_state.last_alert_msg = None

    if alert_msg != st.session_state.last_alert_msg:
        st.session_state.alert_time = now
        st.session_state.last_alert_msg = alert_msg

    elapsed = (now - st.session_state.alert_time).total_seconds()

    if not skip_alerts:
        if elapsed < 10:
            st.toast(alert_msg, icon="⚠️")

# CACHE GLOBAL RISK SCAN
@st.cache_data
def compute_high_risk_districts(_df, _clf_outbreak, model_features):
    results = []
    grps = _df.groupby("district")

    for district in _df["district"].unique():
        temp_df = grps.get_group(district).sort_values("date")

        if temp_df.empty:
            continue

        latest_row = temp_df.iloc[-1]

        try:
            features = latest_row[model_features].values.reshape(1, -1)
            prob = _clf_outbreak.predict_proba(features)[0][1]
            risk_idx = prob * 100

            if risk_idx > 1:
                results.append((district, risk_idx))
        except:
            continue

    return sorted(results, key=lambda x: x[1], reverse=True)

@st.cache_data
def get_high_risk():
    return compute_high_risk_districts(df, clf_outbreak, model_features)

high_risk_districts = get_high_risk()

# NATIONAL ALERT
if high_risk_districts:

    top_district, top_risk = high_risk_districts[0]

    st.error(f"""
    🚨 NATIONAL ALERT: Emerging Health Threat Detected
    High outbreak probability identified in **{top_district} ({top_risk:.2f}%)**
        Review recommended.
    """)

    st.divider()

    st.markdown("### AI Recommendations")
    st.caption("AI-ranked districts requiring immediate attention")

    top_districts = high_risk_districts[:3]
    cols = st.columns(len(top_districts))

    for i, (d, r) in enumerate(top_districts):

        label, level = classify_risk(r / 100)

        color = {
            "CRITICAL": "#5c0000",
            "HIGH": "#7a3e00",
            "MODERATE": "#665c00"
        }.get(level, "#1e5c00")

        with cols[i]:

            st.markdown(f"""
                <div style="
                    background: {color};
                    padding: 20px;
                    border-radius: 12px;
                    color: white;
                    text-align: center;
                    height: 120px;
                    box-shadow: 0 4px 10px rgba(0,0,0,0.3);
                ">
                    <h4 style="margin:0; padding:0;">{d}</h4>
                    <div style="font-size:14px; opacity:0.8;">{label}</div>
                    <div style="font-size:22px; font-weight:bold; margin-top:10px;">{r:.1f}%</div>
                </div>
            """, unsafe_allow_html=True)

            def make_district_click_callback(district_name):
                def callback():
                    st.session_state.pending_district = district_name
                    st.session_state["suppress_alert"] = True
                return callback

            st.button(
                f"View {d}",
                key=f"btn_{d}",
                use_container_width=True,
                help=f"Click to analyze {d}",
                on_click=make_district_click_callback(d)
            )

            if d not in st.session_state.sms_sent_districts:
                def make_sms_callback(district_name, risk_level, risk_val):
                    def callback():
                        st.session_state.pending_sms_district = district_name
                        st.session_state.pending_sms_level = risk_level
                        st.session_state.pending_sms_risk = round(risk_val, 2)
                        st.session_state.pending_sms_msg = (
                            f"⚠️ Outbreak risk detected in {district_name}. "
                            f"Risk Index: {risk_val:.2f}%. "
                            f"Immediate public health response recommended."
                        )
                    return callback

                st.button(
                    f"📲 Alert {d}",
                    key=f"sms_{d}",
                    use_container_width=True,
                    type="primary",
                    on_click=make_sms_callback(d, level, r)
                )
            else:
                st.success("✅ Alerted", icon="📩")

# FIRE THE SMS DIALOG outside the loop
if (
    "pending_sms_district" in st.session_state
    and st.session_state.pending_sms_district is not None
    and st.session_state.pending_sms_district not in st.session_state.sms_sent_districts
):
    confirm_sms_send(
        message=st.session_state.pending_sms_msg,
        district=st.session_state.pending_sms_district,
        level=st.session_state.pending_sms_level
    )

st.divider()

# CHART THEME HELPER
def apply_chart_theme(fig, title=None, height=400):
    fig.update_layout(
        height=height,
        title=dict(
            text=title or fig.layout.title.text or "",
            font=dict(size=15, color="#f0f0f0"),
            x=0.01
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        font=dict(color="#cccccc", size=12),
        legend=dict(
            bgcolor="rgba(0,0,0,0.3)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,
            font=dict(size=11)
        ),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.07)",
            showline=True,
            linecolor="rgba(255,255,255,0.15)",
            tickfont=dict(size=11)
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.07)",
            showline=True,
            linecolor="rgba(255,255,255,0.15)",
            tickfont=dict(size=11)
        ),
        margin=dict(l=10, r=10, t=45, b=10)
    )
    return fig

@st.fragment
def render_trends_climate():

    # Disease Trends — dual line with fill
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=district_df["date"],
        y=district_df["vector_borne_cases"],
        name="Vector-Borne",
        mode="lines",
        line=dict(color="#ff6b6b", width=2),
        fill="tozeroy",
        fillcolor="rgba(255,107,107,0.12)"
    ))
    fig.add_trace(go.Scatter(
        x=district_df["date"],
        y=district_df["waterborne_cases"],
        name="Waterborne",
        mode="lines",
        line=dict(color="#4ecdc4", width=2),
        fill="tozeroy",
        fillcolor="rgba(78,205,196,0.10)"
    ))
    fig = apply_chart_theme(fig, f"🦟 Disease Trends — {selected_district}", height=380)
    st.plotly_chart(fig, use_container_width=True)

    # Risk trend — smoothed with threshold band
    fig2 = go.Figure()
    avg_risk = district_df["risk_roll7"].mean()
    fig2.add_hrect(
        y0=avg_risk * 1.5, y1=district_df["risk_roll7"].max() * 1.1,
        fillcolor="rgba(255,80,80,0.08)", line_width=0,
        annotation_text="Elevated Zone", annotation_position="top left",
        annotation_font=dict(color="rgba(255,150,150,0.7)", size=10)
    )
    fig2.add_trace(go.Scatter(
        x=district_df["date"],
        y=district_df["risk_roll7"],
        name="Smoothed Risk",
        mode="lines",
        line=dict(color="#f9ca24", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(249,202,36,0.10)"
    ))
    fig2 = apply_chart_theme(fig2, "📈 7-Month Smoothed Risk Trend", height=340)
    st.plotly_chart(fig2, use_container_width=True)

    # Climate drivers — dual axis
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=district_df["date"],
        y=district_df["t2m"],
        name="Temperature (°C)",
        mode="lines",
        line=dict(color="#ff9f43", width=2),
        yaxis="y1"
    ))
    fig3.add_trace(go.Bar(
        x=district_df["date"],
        y=district_df["tp"],
        name="Rainfall (mm)",
        marker_color="rgba(84,160,255,0.55)",
        yaxis="y2"
    ))
    fig3.update_layout(
        yaxis=dict(
            title=dict(text="Temp (°C)", font=dict(color="#ff9f43"))
        ),
        yaxis2=dict(
            title=dict(text="Rainfall (mm)", font=dict(color="#54a0ff")),
            overlaying="y",
            side="right"
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0.3)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1
        ),
        barmode="overlay"
    )
    fig3 = apply_chart_theme(fig3, "🌡️ Climate Drivers — Temperature & Rainfall", height=360)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown(
        "<h5 style='margin-bottom:5px;'>Export Data</h5>",
        unsafe_allow_html=True
    )

    col_d1, col_d2, col_d3 = st.columns(3)

    with col_d1:
        st.download_button(
            "Full Dataset",
            district_df.to_csv(index=False),
            f"{selected_district}_full.csv",
            "text/csv"
        )

    with col_d2:
        st.download_button(
            "Last 12 Months",
            district_df.tail(12).to_csv(index=False),
            f"{selected_district}_last12.csv",
            "text/csv"
        )

    with col_d3:
        st.download_button(
            "Outbreak Periods",
            district_df[district_df["outbreak"] == 1].to_csv(index=False),
            f"{selected_district}_outbreaks.csv",
            "text/csv"
        )

@st.fragment
def render_risk_forecast():

    with st.container(border=True):

        st.markdown(
            "<h4 style='margin-bottom:5px;'>Disease Forecast Engine</h4>",
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Predicted Vector Cases", f"{int(pred_vector)}")

        with col2:
            st.metric("Predicted Water Cases", f"{int(pred_water)}")

        st.markdown(
            "<h4 style='margin-bottom:5px;'>Outbreak Probability</h4>",
            unsafe_allow_html=True
        )

        st.progress(float(pred_outbreak_prob))

        risk_label, _ = classify_risk(pred_outbreak_prob)
        st.markdown(f"### **Status: {risk_label}**")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Outbreak Signal")

        # Color bars by outbreak value
        outbreak_colors = [
            "#ff4757" if v == 1 else "#2ed573"
            for v in district_df["outbreak"]
        ]
        fig4 = go.Figure(go.Bar(
            x=district_df["date"],
            y=district_df["outbreak"],
            marker_color=outbreak_colors,
            name="Outbreak"
        ))
        fig4 = apply_chart_theme(fig4, "🚨 Outbreak Signal Over Time", height=340)
        st.plotly_chart(fig4, use_container_width=True)

    with col_b:
        st.subheader("Top Risk Periods")
        top_risk_df = district_df.sort_values(
            "risk_score", ascending=False
        ).head(10)[[
            "date", "vector_borne_cases",
            "waterborne_cases", "risk_score"
        ]]
        st.dataframe(top_risk_df, use_container_width=True)

    @st.cache_data(ttl=3600)
    def get_forecast(lat, lon):
        try:
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": lat,
                "longitude": lon,
                "daily": "temperature_2m_mean,precipitation_sum",
                "forecast_days": 7,
                "timezone": "auto"
            }
            r = requests.get(url, params=params)
            data = r.json()
            forecast_df = pd.DataFrame({
                "date": data["daily"]["time"],
                "temperature": data["daily"]["temperature_2m_mean"],
                "precipitation": data["daily"]["precipitation_sum"]
            })
            forecast_df["date"] = pd.to_datetime(forecast_df["date"])
            return forecast_df
        except Exception as e:
            st.error(f"Forecast API error: {e}")
            return None

    lat = latest["latitude"]
    lon = latest["longitude"]
    forecast_df = get_forecast(lat, lon)

    with st.container(border=True):

        st.markdown("""
            <style>
            .weather-header {
                display: inline-block;
                font-size: 25px;
                font-weight: 600;
                animation: float 2s ease-in-out infinite;
            }
            @keyframes float {
                0% { transform: translateY(0px); }
                50% { transform: translateY(-6px); }
                100% { transform: translateY(0px); }
            }
            </style>
            <div class="weather-header">
                🌦️ 7-Day Climate Intelligence
            </div>
            """,
            unsafe_allow_html=True
        )

        if forecast_df is not None:
            col1, col2 = st.columns(2)

            with col1:
                fig_temp = go.Figure()
                fig_temp.add_trace(go.Scatter(
                    x=forecast_df["date"],
                    y=forecast_df["temperature"],
                    mode="lines+markers",
                    line=dict(color="#ff9f43", width=2.5),
                    marker=dict(size=7, color="#ff9f43", line=dict(width=1, color="white")),
                    fill="tozeroy",
                    fillcolor="rgba(255,159,67,0.12)",
                    name="Temp °C"
                ))
                fig_temp = apply_chart_theme(fig_temp, "🌡️ Temperature Forecast", height=300)
                st.plotly_chart(fig_temp, use_container_width=True)

            with col2:
                fig_rain = go.Figure(go.Bar(
                    x=forecast_df["date"],
                    y=forecast_df["precipitation"],
                    marker=dict(
                        color=forecast_df["precipitation"],
                        colorscale="Blues",
                        showscale=False
                    ),
                    name="Rainfall mm"
                ))
                fig_rain = apply_chart_theme(fig_rain, "🌧️ Rainfall Forecast", height=300)
                st.plotly_chart(fig_rain, use_container_width=True)

        else:
            st.warning("Forecast data unavailable.")

@st.fragment
def render_alerts():

    with st.container(border=True):

        st.markdown(
            "<h4 style='margin-bottom:5px;'>Alert History</h4>",
            unsafe_allow_html=True
        )

        if st.button("Clear Alert History"):
            st.session_state.alert_history = []
            st.session_state.suppress_alert = True
            st.success("Alert history cleared.")

        if st.session_state.alert_history:
            history_df = pd.DataFrame(st.session_state.alert_history)
            history_df = history_df.sort_values("time", ascending=False)
            st.dataframe(history_df, use_container_width=True)
        else:
            st.info("No alerts triggered yet.")

@st.fragment
def render_risk_map():

    import folium
    from streamlit_folium import st_folium
    from folium.plugins import Fullscreen

    st.markdown(
        "<h4 style='margin-bottom:5px;'>Spatial Hotspot Intelligence</h4>",
        unsafe_allow_html=True
    )

    map_view = st.radio(
        "Select Map View",
        ["Current Hotspot", "Spatiotemporal Timeline"],
        horizontal=True
    )

    status_colors = {
        "Old (Persistent)": "#8B0000",
        "Emerging": "#FF4500",
        "New": "#FFD700",
        "Normal": "#1E90FF"
    }

    legend_html = f'''
    <div style="position: fixed; bottom: 50px; left: 50px; width: auto; min-width: 180px;
                background-color: white; border:2px solid grey; z-index:9999; font-size:12px;
                padding: 10px; color: black; box-shadow: 2px 2px 5px rgba(0,0,0,0.2); border-radius: 5px;">
        <b style="display: block; margin-bottom: 8px; border-bottom: 1px solid #ccc;">
        Hotspot Status (Relative Risk)
        </b>
        <table style="border-spacing: 0 5px;">
            <tr><td><div style="background:{status_colors['Old (Persistent)']}; width:16px; height:16px; border-radius:50%;"></div></td>
                <td style="padding-left:8px;">Old (Persistent)</td></tr>
            <tr><td><div style="background:{status_colors['Emerging']}; width:16px; height:16px; border-radius:50%;"></div></td>
                <td style="padding-left:8px;">Emerging</td></tr>
            <tr><td><div style="background:{status_colors['New']}; width:16px; height:16px; border-radius:50%;"></div></td>
                <td style="padding-left:8px;">New Detected</td></tr>
            <tr><td><div style="background:{status_colors['Normal']}; width:16px; height:16px; border-radius:50%;"></div></td>
                <td style="padding-left:8px;">Normal Risk</td></tr>
        </table>
    </div>
    '''

    if map_view == "Current Hotspot":

        with open(geo_path, "r", encoding="utf-8") as f:
            geo = json.load(f)

        map_df = df[[
            "district", "date", "vector_borne_cases",
            "waterborne_cases", "latitude", "longitude"
        ]].copy()

        map_df["period"] = pd.to_datetime(map_df["date"]).dt.to_period("M")

        monthly_map = map_df.groupby(
            ["district", "period"], as_index=False
        ).agg({
            "vector_borne_cases": "mean",
            "waterborne_cases": "mean",
            "latitude": "first",
            "longitude": "first"
        })

        monthly_map["risk_score"] = (
            0.6 * monthly_map["vector_borne_cases"]
        ) + (
            0.4 * monthly_map["waterborne_cases"]
        )

        all_periods = sorted(monthly_map["period"].unique())
        curr_p, prev_p = all_periods[-1], all_periods[-2]

        comparison = monthly_map[
            monthly_map["period"] == curr_p
        ].merge(
            monthly_map[monthly_map["period"] == prev_p][[
                "district", "risk_score"
            ]],
            on="district",
            suffixes=('', '_prev')
        )

        threshold = monthly_map["risk_score"].quantile(0.80)

        def classify_hotspot(row):
            is_curr_high = row['risk_score'] > threshold
            is_prev_high = row['risk_score_prev'] > threshold
            if is_curr_high and is_prev_high:
                return "Old (Persistent)"
            if is_curr_high and not is_prev_high:
                return "Emerging"
            if is_curr_high:
                return "New"
            return "Normal"

        comparison["status"] = comparison.apply(classify_hotspot, axis=1)

        full_map = monthly_map[
            monthly_map["period"] == curr_p
        ].merge(
            comparison[["district", "status"]],
            on="district",
            how="left"
        ).fillna("Normal")

        target = full_map[full_map["district"] == selected_district]
        map_center = (
            [target["latitude"].iloc[0], target["longitude"].iloc[0]]
            if not target.empty else [1.37, 32.3]
        )
        map_zoom = 12 if not target.empty else 7

        m = folium.Map(
            location=map_center,
            zoom_start=map_zoom,
            tiles=None,
            control_scale=True
        )

        folium.TileLayer('OpenStreetMap', name='OpenStreetMap').add_to(m)
        folium.TileLayer(
            tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
            attr='Google',
            name='Satellite Hybrid',
            overlay=False
        ).add_to(m)

        pulse_css = """
        <style>
        .pulse {
            width: 14px;
            height: 14px;
            border-radius: 50%;
            background: #FF4500;
            box-shadow: 0 0 0 rgba(255,69,0, 0.7);
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(255,69,0, 0.7); }
            70% { transform: scale(1.4); box-shadow: 0 0 0 15px rgba(255,69,0, 0); }
            100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(255,69,0, 0); }
        }
        </style>
        """
        m.get_root().html.add_child(folium.Element(pulse_css))

        for _, row in full_map.iterrows():
            color = status_colors.get(row['status'], "gray")

            if row['status'] != "Normal":
                folium.CircleMarker(
                    location=[row["latitude"], row["longitude"]],
                    radius=14, color=color, fill=True,
                    fill_opacity=0.15, weight=0
                ).add_to(m)

            if row['status'] == "Emerging":
                folium.Marker(
                    location=[row["latitude"], row["longitude"]],
                    icon=folium.DivIcon(html='<div class="pulse"></div>')
                ).add_to(m)

            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=7, color="white", weight=1,
                fill_color=color, fill=True, fill_opacity=0.9,
                tooltip=f"<b>{row['district']}</b>",
                popup=f"<b>{row['district']}</b><br>Risk: {row['risk_score']:.2f}<br>Status: {row['status']}"
            ).add_to(m)

        m.get_root().html.add_child(folium.Element(legend_html))
        folium.LayerControl().add_to(m)
        Fullscreen().add_to(m)

        st_folium(
            m,
            width="stretch",
            height=900,
            key=f"int_map_{selected_district}",
            returned_objects=[]
        )

    else:
        st.info("Dynamic Hotspot Evolution")

        map_df_anim = df.copy()
        map_df_anim["Month"] = map_df_anim["date"].dt.strftime('%Y-%m')

        monthly_avg = map_df_anim.groupby(
            ["district", "Month"], as_index=False
        ).agg({
            "vector_borne_cases": "mean",
            "waterborne_cases": "mean",
            "latitude": "first",
            "longitude": "first"
        })

        monthly_avg["risk_score"] = (
            0.6 * monthly_avg["vector_borne_cases"] +
            0.4 * monthly_avg["waterborne_cases"]
        )
        monthly_avg = monthly_avg.sort_values(["district", "Month"])

        threshold = monthly_avg["risk_score"].quantile(0.80)
        monthly_avg["is_high"] = monthly_avg["risk_score"] > threshold
        monthly_avg["prev_high"] = (
            monthly_avg.groupby("district")["is_high"]
            .shift(1).fillna(False)
        )

        def map_legend_logic(row):
            if row["is_high"] and row["prev_high"]:
                return "Old (Persistent)"
            elif row["is_high"] and not row["prev_high"]:
                return "Emerging"
            elif not row["is_high"] and row["prev_high"]:
                return "Declining"
            else:
                return "Normal"

        monthly_avg["status"] = monthly_avg.apply(map_legend_logic, axis=1)

        statuses = ["Old (Persistent)", "Emerging", "Declining", "Normal"]
        months = monthly_avg["Month"].unique()

        ghost_frames = []
        for mo in months:
            for s in statuses:
                ghost_frames.append({
                    "district": "GHOST", "Month": mo, "status": s,
                    "latitude": monthly_avg["latitude"].mean(),
                    "longitude": monthly_avg["longitude"].mean(),
                    "risk_score": 0.0001
                })

        ghost_df = pd.DataFrame(ghost_frames)
        stable_df = pd.concat(
            [monthly_avg, ghost_df], ignore_index=True
        ).sort_values("Month")

        color_map = {
            "Old (Persistent)": "#8B0000",
            "Emerging": "#FF4500",
            "Declining": "#FFD700",
            "Normal": "#1E90FF"
        }

        fig = px.scatter_mapbox(
            stable_df,
            lat="latitude", lon="longitude",
            size="risk_score", color="status",
            animation_frame="Month", animation_group="district",
            hover_name="district",
            color_discrete_map=color_map,
            category_orders={"status": statuses},
            size_max=25, height=900
        )

        fig.update_layout(
            mapbox_style="open-street-map",
            margin=dict(l=0, r=0, t=0, b=0),
            mapbox=dict(
                uirevision='constant',
                center=dict(
                    lat=monthly_avg["latitude"].mean(),
                    lon=monthly_avg["longitude"].mean()
                ),
                zoom=6
            ),
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 600, "redraw": True}, "fromcurrent": True}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ]
            }]
        )

        fig.update_traces(selector=dict(name="GHOST"), visible=False)
        st.plotly_chart(fig, use_container_width=True)


# NAVIGATION
tab1, tab2, tab3, tab4 = st.tabs([
    "ANALYSIS TRENDS ",
    "FORECAST INTELLIGENCE ",
    "RISK MAPS ",
    "ALERTS "
])

with tab1:
    render_trends_climate()

with tab2:
    render_risk_forecast()

with tab3:
    render_risk_map()

with tab4:
    render_alerts()


# FOOTER
st.caption(
    "GeoAI Early Warning System | ERA5-Land and Synthetic Epidemiology Engine"
)
