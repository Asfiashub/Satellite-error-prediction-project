import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import io
import os 
from scipy import stats 
import tensorflow as tf 
from functools import lru_cache 
from satellite_predictor import SatelliteErrorPredictor
from utils.visualization import (
    make_globe_placeholder,
    create_error_timeline,
    create_3d_error_viz,
    create_heatmap_correlation,
    create_prediction_comparison
)

try:
    import xarray as xr
    HAS_XARRAY = True
except Exception:
    HAS_XARRAY = False

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Satellite Error Prediction",
    page_icon="🛰️",
    layout="wide",
)

# ===============================
# CUSTOM CSS
# ===============================
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #0a0e27, #1a1f3a); 
    color: white; 
}
h1, h2, h3 { 
    color: #00d4ff; 
    text-align:center; 
    text-shadow: 0 0 10px #00d4ff; 
}
.sidebar .sidebar-content { 
    background: #000020; 
    color: #00ffff;
}
.stButton>button {
    background: linear-gradient(135deg, #00d4ff, #9d4edd);
    color: white; border: none; border-radius: 8px; font-weight: bold;
    box-shadow: 0 0 20px rgba(0,212,255,0.5);
}
.stButton>button:hover { 
    transform: scale(1.05); 
}
</style>
""", unsafe_allow_html=True)

# ===============================================
# DATA UTILITY FUNCTIONS
# ===============================================

@st.cache_resource
def load_predictor():
    predictor = SatelliteErrorPredictor(sequence_length=7)
    predictor.load_models("models")
    return predictor


@st.cache_data
def normalize_columns(df):
    df = df.copy()

    df.columns = [c.lower().strip().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_") for c in df.columns]
    
    col_map = {
        'satclockerror_m': 'clock_error_m',
        'satclockerror__m': 'clock_error_m',
        'ephemeris_error': 'ephemeris_error_m',

        'x_error': 'x_error_m',
        'x_error_m': 'x_error_m',
        'x_error__m': 'x_error_m',

        'y_error': 'y_error_m',
        'y_error_m': 'y_error_m',
        'y_error__m': 'y_error_m',

        'z_error': 'z_error_m',
        'z_error_m': 'z_error_m',
        'z_error__m': 'z_error_m',
    }
    
    df.rename(columns=col_map, inplace=True)
    
    if 'utc_time' in df.columns:
        df["utc_time"] = pd.to_datetime(df["utc_time"], errors="coerce")

    required_cols = ["clock_error_m", "x_error_m", "y_error_m", "z_error_m"]

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        st.error(f"❌ Missing required columns: {missing}")
        st.stop()

    if 'satname' not in df.columns:
        df['satname'] = "SAT-1"

    if 'ephemeris_error_m' not in df.columns:
        df['ephemeris_error_m'] = np.sqrt(
            df['x_error_m']**2 + df['y_error_m']**2 + df['z_error_m']**2
        )

    return df


@st.cache_resource
def load_data_files():
    files = [
        "DATA_GEO_Train",
        "DATA_MEO_Train",
        "DATA_MEO_Train2"
    ]

    data = []
    for base in files:
        parquet_file = f"{base}.parquet"
        csv_file = f"{base}.csv"
        
        if os.path.exists(parquet_file):
            df = pd.read_parquet(parquet_file)
            df = normalize_columns(df)
            data.append((parquet_file, df))
        elif os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            try:
                df.to_parquet(parquet_file)
            except Exception:
                pass
            df = normalize_columns(df)
            data.append((csv_file, df))

    if not data:
        st.error("No data files found")
        st.stop()

    return data


if 'data_files' not in st.session_state:
    st.session_state['data_files'] = load_data_files()

# ===============================================
# SIDEBAR
# ===============================================
page = st.sidebar.radio(
    "Go to",
    [
        "Dashboard",
        "Data Analysis",
        "3D Visualization",
        "Predictions",
        "About"
    ]
)

# ===============================================
# DASHBOARD
# ===============================================
if page == "Dashboard":
    st.title("🛰️ Satellite Error Dashboard")

    st.subheader("Satellite Constellation Overview")
    globe = make_globe_placeholder()

    if type(globe).__name__ == "Deck":
        st.pydeck_chart(globe)
    else:
        st.plotly_chart(globe, use_container_width=True)

    for name, df in st.session_state['data_files']:
        st.markdown(f"### 📊 Data Analysis: `{name}`")
        st.write(f"Shape: {df.shape}")
        
        viz_df = df.copy()
        viz_df.rename(columns={
            "clock_error_m": "satclockerror (m)",
            "x_error_m": "x_error (m)",
            "y_error_m": "y_error (m)",
            "z_error_m": "z_error (m)"
        }, inplace=True)
        
        st.plotly_chart(create_error_timeline(viz_df, f"Timeline - {name}"), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_heatmap_correlation(viz_df), use_container_width=True)
        with col2:
            st.plotly_chart(create_3d_error_viz(viz_df, f"3D Error Path"), use_container_width=True)
            
        st.markdown("---")


elif page == "Data Analysis":

    st.title("📊 Data Analysis")

    df = st.session_state['data_files'][0][1]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    selected_col = st.selectbox("Select Parameter", numeric_cols)

    st.plotly_chart(
        px.line(df, x="utc_time", y=selected_col, title=f"{selected_col} Trend"),
        use_container_width=True
    )

    st.subheader("Statistics")

    stats_df = df[numeric_cols].describe().T
    stats_df["skewness"] = df[numeric_cols].skew()
    stats_df["kurtosis"] = df[numeric_cols].kurt()

    st.dataframe(stats_df)

    st.subheader("Correlation Heatmap")

    corr = df[numeric_cols].corr()

    fig = px.imshow(corr, text_auto=True, aspect="auto")

    st.plotly_chart(fig, use_container_width=True)


elif page == "3D Visualization":

    st.title("🌍 3D Visualization")

    df = st.session_state['data_files'][0][1]

    viz_df = df.copy()
    viz_df.rename(
        columns={
            "clock_error_m": "satclockerror (m)",
            "x_error_m": "x_error (m)",
            "y_error_m": "y_error (m)",
            "z_error_m": "z_error (m)"
        },
        inplace=True
    )

    fig = create_3d_error_viz(viz_df, "Satellite Error Path")

    st.plotly_chart(fig, use_container_width=True)

    try:
        img = fig.to_image(format="png")

        st.download_button(
            "⬇ Download PNG",
            img,
            "visualization.png",
            "image/png"
        )
    except:
        pass


# ===============================================
# PREDICTIONS
# ===============================================
elif page == "Predictions":

    st.title("🤖 Prediction Panel")

    df = st.session_state['data_files'][0][1]

    sat_type = st.selectbox(
        "Satellite Type",
        ["GEO/GSO", "MEO"],
        key="satellite_type"
    )

    if st.button("Run Prediction", key="run_prediction_btn"):

        predictor = load_predictor()

        orbit_type = "GEO" if sat_type == "GEO/GSO" else "MEO"

        prediction_df = df[["utc_time", "clock_error_m", "x_error_m", "y_error_m", "z_error_m"]].copy()

        prediction_df.rename(
            columns={
                "clock_error_m": "satclockerror (m)",
                "x_error_m": "x_error (m)",
                "y_error_m": "y_error (m)",
                "z_error_m": "z_error (m)"
            },
            inplace=True
        )

        predictions = predictor.predict_8th_day(prediction_df, orbit_type)

        st.success("Prediction completed")
        st.json(predictions)

        csv = pd.DataFrame([predictions]).to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇ Download Prediction CSV",
            csv,
            "prediction.csv",
            "text/csv"
        )

        st.subheader("Prediction Visualization")

        cols = st.columns(2)
        viz_idx = 0

        for k, v in predictions.items():

            hist_col = k
            if k == "satclockerror (m)":
                hist_col = "clock_error_m"
            elif k == "x_error (m)":
                hist_col = "x_error_m"
            elif k == "y_error (m)":
                hist_col = "y_error_m"
            elif k == "z_error (m)":
                hist_col = "z_error_m"

            if hist_col in df.columns:
                hist_series = df[hist_col].tail(14).reset_index(drop=True)

                fig = create_prediction_comparison(hist_series, v, k)

                with cols[viz_idx % 2]:
                    st.plotly_chart(fig, use_container_width=True)

                viz_idx += 1


# ===============================================
# ABOUT
# ===============================================
elif page == "About":
    st.header("About Project")
    st.markdown("---")
    st.markdown("""
    <h2 style="text-align:left;">Satellite error prediction</h2>

    <p>
    🛰️ <b></b> A solution designed for <b>ISRO</b> Predicting Time varying satellite error patterns. 
    This project integrates <b>advanced ML architectures</b> with <b>physics-based modeling</b> to estimate and forecast deviations between broadcast GNSS values and ICD-modeled parameters — enabling precise orbit and clock error predictions for future epochs.
    </p>

    <ul>
      <li> <b>Data:</b> Seven-day GNSS dataset containing recorded clock and ephemeris discrepancies for GEO/GSO and MEO satellites.  
      The platform supports flexible data uploads — users can input datasets of varying durations for different satellites.</li>

      <li> <b>Forecast Horizons:</b> Configurable validity windows ranging from 15 minutes to 24 hours (15 min, 30 min, 1 hr … 24 hr), aligning with ISRO’s accuracy requirements.</li>

      <li> <b>Model Architectures Supported</b> (with more coming soon):
        <ul>
          <li> RNNs (LSTM, GRU) — sequential error forecasting</li>
          <li> GANs — synthetic data augmentation for low-error epochs</li>
          <li> Transformers — capturing long-range temporal dependencies</li>
          <li> Gaussian Processes — probabilistic uncertainty quantification</li>
        </ul>
      </li>

      <li> <b>Statistical Analysis:</b> Includes residual diagnostics and Shapiro–Wilk normality testing to assess distributional characteristics.</li>
    </ul>

    <p>
     <b></b>
    </p>
    """, unsafe_allow_html=True)