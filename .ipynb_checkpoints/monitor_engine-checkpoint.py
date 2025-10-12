import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Set Pandas display options (updated to support enough elements)
pd.set_option("styler.render.max_elements", 600000)

# Load the trained LightGBM model
model_path = os.path.join(os.path.dirname(__file__), 'lightgbm_model.pkl')
model = joblib.load(model_path)

# Set page configuration
st.set_page_config(page_title="Engine Health Dashboard", layout="wide")

# Title and description
st.title("Engine Health Monitoring Dashboard")
st.markdown("Monitor, assess, and predict the health of turbofan engines using NASA CMAPSS data.")

# Sidebar inputs
st.sidebar.header("Configuration")
threshold = st.sidebar.slider("RUL Threshold (for Inspection)", 10, 70, 30)
uploaded_file = st.sidebar.file_uploader("Upload Engine Sensor CSV", type=["csv"])

# Main logic
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Preview dataframe
    st.write("Uploaded Data Preview:")
    st.dataframe(df.head())

    # Extract and save engine IDs
    engine_ids = df['engine_id'].unique()

    # Prepare prediction features
    df_predict = df.copy()
    features_to_drop = ['engine_id', 'op_set_1', 'op_set_2', 'op_set_3', 'T2', 'P2', 'P15', 'epr', 'farB', 'Nf_dmd', 'PCNfR_dmd']
    features = [col for col in df_predict.columns if col not in features_to_drop]

    sensor_columns = [col for col in features if col != 'cycle_time']
    selected_sensor = st.sidebar.selectbox("Sensor to Monitor", sensor_columns)

    # Predict RUL
    df_predict['predicted_rul'] = model.predict(df_predict[features])

    # Categorize engine health status
    df_predict['action'] = df_predict['predicted_rul'].apply(
        lambda x: 'Repair' if x <= 15 else ('Inspect' if x <= threshold else 'Monitor')
    )

    # Metrics overview
    # Metrics overview (grouped by engine_id)
    engines_repair = df_predict[df_predict["action"] == "Repair"]["engine_id"].nunique()
    engines_inspect = df_predict[df_predict["action"] == "Inspect"]["engine_id"].nunique()

    st.metric("Engines Needing Repair", engines_repair)
    st.metric("Engines Needing Inspection", engines_inspect)


    # Highlight function for DataFrame
    def highlight_action(val):
        colors = {"Repair": "#FFCCCC", "Inspect": "#FFF5CC", "Monitor": "#CCFFCC"}
        return f'background-color: {colors.get(val, "")}'

    # Health table — show only top 200 for styling
    st.header("Engine Health Overview")
    top_n = 200
    subset_df = df_predict.sort_values("predicted_rul").head(top_n)
    st.dataframe(
        subset_df.style.applymap(highlight_action, subset=["action"]),
        use_container_width=True
    )

    # Histogram of risk
    # Histogram of risk (one row per engine)
    df_latest = (
        df_predict.sort_values("cycle_time")
        .groupby("engine_id", as_index=False)
        .last()
        )

    st.subheader("Risk Assessment")
    fig1, ax1 = plt.subplots()
    sns.histplot(data=df_latest, x="predicted_rul", hue="action", multiple="stack", palette="muted", ax=ax1)
    st.pyplot(fig1)

    # Engine-specific diagnostics
    selected_engine = st.selectbox("Drill-down: Select Engine ID", engine_ids)

    # Filter and plot selected engine's sensor trend
    engine_df = df[df.engine_id == selected_engine].sort_values("cycle_time")
    if selected_sensor in sensor_columns:
        sensor_data = engine_df[["cycle_time", selected_sensor]]

        st.subheader(f"{selected_sensor} Over Time for Engine {selected_engine}")
        fig2, ax2 = plt.subplots()
        sns.lineplot(data=sensor_data, x="cycle_time", y=selected_sensor, ax=ax2)
        st.pyplot(fig2)
    else:
        st.warning(f"{selected_sensor} not found for Engine {selected_engine}")

    # Maintenance recommendation
    latest_rul = df_latest[df_latest['engine_id'] == selected_engine]['predicted_rul'].values[0]


    if latest_rul <= 15:
        st.error(f"Engine {selected_engine} should be repaired immediately! RUL: {latest_rul}")
    elif latest_rul <= threshold:
        st.warning(f"Engine {selected_engine} needs inspection soon. RUL: {latest_rul}")
    else:
        st.success(f"Engine {selected_engine} is operating normally. RUL: {latest_rul}")

else:
    st.info("Please upload a CSV file to begin analysis.")
