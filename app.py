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
    page_title="Satellite Position Error and Satellite Clock Error Prediction using LSTM Networks on GEO and MEO GNSS Data.",
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
# VISUALIZATION HELPERS (CACHED)

@st.cache_data(show_spinner="Generating 3D Visualization...")
def create_3d_globe(df, sample_rate=10):
    """Create rotating 3D globe with satellite error points. CACHED for performance."""
    if len(df) > 1000:
         sample_rate = max(10, len(df) // 1000)
         
    phi, theta = np.mgrid[0:np.pi:60j, 0:2*np.pi:120j]
    xs, ys, zs = np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)

    df = df.copy()
    
    if not all(c in df.columns for c in ['x_error_m', 'y_error_m', 'z_error_m']):
        df['magnitude'] = 1.0 
    else:
        df["magnitude"] = np.sqrt(df["x_error_m"]**2 + df["y_error_m"]**2 + df["z_error_m"]**2)

    df["scale"] = 1 + (df["magnitude"] / (df["magnitude"].max() + 1e-6)) * 0.3
    
    angles = np.linspace(0, 2*np.pi * (len(df)/200), len(df))
    df["x_pos"] = df["scale"] * np.cos(angles) * np.cos(angles * 0.1)
    df["y_pos"] = df["scale"] * np.sin(angles) * np.cos(angles * 0.1)
    df["z_pos"] = np.sin(np.linspace(-np.pi/2, np.pi/2, len(df)))
    
    df_sample = df.iloc[::sample_rate]

    fig = go.Figure()
    fig.add_trace(go.Surface(x=xs, y=ys, z=zs, colorscale="Earth", showscale=False, opacity=0.9))
    fig.add_trace(go.Scatter3d(
        x=df_sample["x_pos"],
        y=df_sample["y_pos"],
        z=df_sample["z_pos"],
        mode="markers",
        marker=dict(size=df_sample["magnitude"].apply(lambda x: max(3, x*3/df_sample["magnitude"].max())), 
                    color=df_sample["magnitude"], colorscale="Plasma", showscale=True),
        hovertext=df_sample["satname"],
        name="Satellite Error Position"
    ))
    
    fig.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False),
                                 zaxis=dict(visible=False), bgcolor="black"),
                      margin=dict(l=0, r=0, t=30, b=0), height=500,
                      coloraxis_colorbar=dict(title="Error Magnitude"),
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
    return fig

# --- Plotting Helpers ---
@st.cache_data(show_spinner="Generating Time-Series Plot...")
def create_time_series_plot(df_filtered, y_col, sat):
    return px.line(df_filtered, x="utc_time", y=y_col, 
                    title=f"Trend for {y_col.replace('_', ' ').title()} for {sat}", height=400,
                    template="plotly_dark")

@st.cache_data(show_spinner="Generating Correlation Heatmap...")
def create_correlation_heatmap(df_filtered, numeric_cols):
    corr_matrix = df_filtered[numeric_cols].corr()
    return px.imshow(corr_matrix, text_auto=True, aspect="auto", 
                     color_continuous_scale=px.colors.diverging.RdBu,
                     title="Correlation Matrix of Error Parameters",
                     template="plotly_dark")

@st.cache_data(show_spinner="Generating Comparison Plot...")
def create_comparison_plot(comparison_results, selected_models, horizon):
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(x=comparison_results["utc_time"], y=comparison_results["Actual"], 
                                  name="Actual Error", line=dict(color="#00d4ff", width=3)))
    
    colors = px.colors.qualitative.Vivid
    for i, model in enumerate(selected_models):
        col_name = f'Predicted ({model})'
        fig_pred.add_trace(go.Scatter(x=comparison_results["utc_time"], y=comparison_results[col_name], 
                                      name=col_name, line=dict(color=colors[i % len(colors)], dash='dash')))

    fig_pred.update_layout(title=f"Multi-Model Clock Error Comparison ({horizon} Horizon)", 
                           xaxis_title="Time", yaxis_title="Clock Error (m)", template="plotly_dark")
    return fig_pred

@st.cache_data(show_spinner="Generating Residual Plots...")
def create_residual_plots(residuals):
    fig_res = go.Figure(go.Histogram(x=residuals, nbinsx=50, name='Residuals', marker_color='#9d4edd'))
    fig_res.update_layout(xaxis_title="Residual Error (m)", yaxis_title="Count", template="plotly_dark")
    
    qq_fig = go.Figure()
    residuals_clean = residuals[~np.isnan(residuals)]
    if len(residuals_clean) > 0:
        (osm, osr), (slope, intercept, r) = stats.probplot(residuals_clean, dist='norm', fit=True)
        qq_fig.add_trace(go.Scatter(x=osm, y=osr, mode='markers', name='Residuals', marker=dict(color="#00d4ff")))
        qq_fig.add_trace(go.Line(x=osm, y=intercept + slope*np.array(osm), name='Normal Fit', line=dict(color='red', dash='dash')))
        qq_fig.update_layout(xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles", template="plotly_dark")
    else:
        qq_fig.update_layout(title="Not enough valid data for Q-Q Plot")
        
    return fig_res, qq_fig

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
    st.markdown("""
        <div style="text-align:center; padding:30px; border-radius:15px;
                    background: linear-gradient(135deg, #0a0e27, #1a1f3a);
                    box-shadow: 0 0 20px #00d4ff; margin-bottom:20px;">
            <h1 style="font-size:48px; color:#00d4ff; font-family:Orbitron;
                        text-shadow:0 0 15px #00d4ff;"> SATELLITE ERROR PREDICTION SYSTEM</h1>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.write("Next-generation system for analyzing and predicting satellite ephemeris & clock errors.")

    st.markdown("---")

    st.subheader("📁 Data Upload & Summary")
    uploaded_files = st.file_uploader(
        "Drag & drop new CSV files or click to browse",
        type=["csv"], accept_multiple_files=True
    )
    
    if uploaded_files:
        new_data_files = []
        for uploaded_file in uploaded_files:
            try:
                df_uploaded = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode("utf-8")))
                df_uploaded = normalize_columns(df_uploaded) 
                new_data_files.append((uploaded_file.name, df_uploaded))
            except Exception as e:
                st.warning(f"⚠️ Could not read {uploaded_file.name}. Error: {e}")

        if new_data_files:
             st.cache_resource.clear()
             st.session_state['data_files'].extend(new_data_files)
             st.session_state['df_main'] = st.session_state['data_files'][0][1] 
             
             total_uploaded_rows = sum(len(df) for name, df in st.session_state['data_files'])
             st.success(f"✅ Uploaded **{len(new_data_files)}** new file(s). Total rows loaded: **{total_uploaded_rows}**. Metrics updating...")

             st.rerun() 
    
    st.markdown("---") 

    # --- DATA OVERVIEW & METRICS ---
    if st.session_state['data_files']:
        st.subheader("Loaded Data Overview")
        
        sat_count = 0
        total_rows = 0
        last_time = pd.NaT

        for file_name, df in st.session_state['data_files']:
            if 'satname' in df.columns:
                sat_count += df['satname'].nunique()
            total_rows += len(df)
            
            if 'utc_time' in df.columns and pd.notna(df['utc_time'].max()):
                current_max = df['utc_time'].max()
                if pd.isna(last_time) or current_max > last_time:
                    last_time = current_max

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Satellites", sat_count)
        c2.metric("Total Data Points", total_rows)
        c3.metric("Last Data Timestamp", str(last_time.date()) if pd.notna(last_time) else "N/A")
        
        st.markdown("---")

        st.info("Available Data Files (Must be present in script directory or uploaded):")
        for file_name, df in st.session_state['data_files']:
             status = "REAL DATA"
             
             start_time = df['utc_time'].min() if 'utc_time' in df.columns else None
             end_time = df['utc_time'].max() if 'utc_time' in df.columns else None
             start_str = str(start_time.date()) if pd.notna(start_time) else "N/A"
             end_str = str(end_time.date()) if pd.notna(end_time) else "N/A"
             
             st.markdown(f"- **{file_name}** ({status}): {len(df)} rows from {start_str} to {end_str}")
             
        st.markdown("---")
        
        # --- 3D Visualization (CACHED) ---
        st.subheader(f"🌍 Global Error Visualization (using {st.session_state['data_files'][0][0]})")
        
        if 'df_main' in st.session_state and not st.session_state['df_main'].empty:
             st.plotly_chart(create_3d_globe(st.session_state['df_main']), use_container_width=True)
        else:
             st.warning("Cannot display 3D visualization. Main data frame is empty or columns are missing.")
    
    else:
        st.warning("No data files are currently loaded.")

elif page != "About":
    
    if not st.session_state['data_files']:
        st.error("🚨 No data is loaded. Please go to the **Dashboard** to upload a file or check the file names.")
        if st.button("Reload Data"):
            st.rerun()
        st.stop()
        
    st.subheader(f"Data Source Selector: ")
    data_options = [name for name, df in st.session_state['data_files']]
    selected_data_name = st.selectbox("Select Dataset to Analyze", data_options)
    
    selected_df = next(df for name, df in st.session_state['data_files'] if name == selected_data_name).copy()
    selected_df = normalize_columns(selected_df) 
    
    st.markdown("---")

    # --- DATA ANALYSIS PAGE ---
    if page == "Data Analysis":
        st.header("📊 Data Exploration & Correlation")
        st.markdown("---")

        col_sat, col_param = st.columns(2)
        sat_options = selected_df["satname"].unique() if 'satname' in selected_df.columns else ["SAT-1"]
        sat = col_sat.selectbox("Select Satellite", sat_options)
        df_sat = selected_df[selected_df["satname"] == sat].copy() if 'satname' in selected_df.columns else selected_df.copy()

        numeric_cols = df_sat.select_dtypes(include=[np.number]).columns.tolist()
        
        default_y_col = 'clock_error_m' if 'clock_error_m' in numeric_cols else numeric_cols[0] if numeric_cols else None
        
        if default_y_col is None:
            st.error("No numeric columns available for analysis.")
            st.stop()
            
        y_col = col_param.selectbox("Select Time-Series Parameter", numeric_cols, index=numeric_cols.index(default_y_col))

        time_min, time_max = df_sat['utc_time'].min(), df_sat['utc_time'].max()
        if pd.isna(time_min) or pd.isna(time_max):
            st.warning("Time data is invalid or missing for filtering.")
            df_filtered = df_sat
        else:
            time_range = st.slider(
                'Zoom Time Range', 
                min_value=time_min.to_pydatetime(), 
                max_value=time_max.to_pydatetime(), 
                value=(time_min.to_pydatetime(), time_max.to_pydatetime()), 
                format="YYYY-MM-DD HH:mm"
            )
            df_filtered = df_sat[
                (df_sat['utc_time'] >= time_range[0]) & (df_sat['utc_time'] <= time_range[1])
            ]
        
        st.subheader(f"Time-Series: {y_col.replace('_', ' ').title()} for {sat}")
        st.plotly_chart(create_time_series_plot(df_filtered, y_col, sat), use_container_width=True)
        
        st.subheader("Statistics Panel")
        desc = df_filtered[numeric_cols].describe().T
        desc['skewness'] = df_filtered[numeric_cols].skew()
        desc['kurtosis'] = df_filtered[numeric_cols].kurt()
        st.dataframe(desc.style.format('{:.4f}'))
        
        st.subheader("Correlation Heatmap")
        st.plotly_chart(create_correlation_heatmap(df_filtered, numeric_cols), use_container_width=True)


    # --- 3D VISUALIZATION PAGE ---
    elif page == "3D Visualization":
        st.header("🌐 3D Earth / Satellite Visualization")
        st.markdown("---")
        st.markdown("### Error Magnitude and Simulated Orbit")
        st.plotly_chart(create_3d_globe(selected_df), use_container_width=True)
        st.markdown("""
        <p style='text-align:center; color:gray;'>
        Points are color-coded and sized by total error magnitude. The orbital paths are simulated.
        </p>
        """, unsafe_allow_html=True)




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
