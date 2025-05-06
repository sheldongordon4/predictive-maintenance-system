"""
# Create a template Streamlit app that reflects all best practices for usability and relevance

streamlit_template = """
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(page_title="Engine Health Dashboard", layout="wide")

# Title
st.title("Engine Health Monitoring Dashboard")
st.markdown("Monitor, assess, and predict the health of turbofan engines using NASA CMAPSS data.")

# Sidebar filters
st.sidebar.header("🔧 Configuration")
threshold = st.sidebar.slider("RUL Threshold (for Inspection)", 10, 50, 30)
selected_sensor = st.sidebar.selectbox("Sensor to Monitor", [f"sensor_{i}" for i in range(1, 22)])

# Simulated engine data
engine_ids = list(range(1, 101))
np.random.seed(42)
engine_df = pd.DataFrame({
    "engine_id": engine_ids,
    "latest_cycle": np.random.randint(80, 150, size=100),
    "predicted_rul": np.random.randint(5, 100, size=100)
})
engine_df["action"] = engine_df["predicted_rul"].apply(
    lambda x: "Replace" if x <= 15 else ("Inspect" if x <= threshold else "Monitor"))

# Engine Health Table
st.header("Engine Health Overview")
st.dataframe(engine_df.sort_values("predicted_rul"), use_container_width=True)

# Risk Distribution Plot
st.subheader("Risk Assessment")
fig1, ax1 = plt.subplots()
sns.histplot(data=engine_df, x="predicted_rul", hue="action", multiple="stack", palette="muted", ax=ax1)
st.pyplot(fig1)

# Select engine
selected_engine = st.selectbox("Drill-down: Select Engine ID", engine_ids)
sensor_data = pd.DataFrame({
    "cycle": np.arange(1, engine_df.loc[engine_df.engine_id == selected_engine, "latest_cycle"].values[0] + 1),
    selected_sensor: np.random.normal(loc=100, scale=10, size=engine_df.loc[engine_df.engine_id == selected_engine, "latest_cycle"].values[0])
})

# Sensor Trend Plot
st.subheader(f"{selected_sensor} Over Time for Engine {selected_engine}")
fig2, ax2 = plt.subplots()
sns.lineplot(data=sensor_data, x="cycle", y=selected_sensor, ax=ax2)
st.pyplot(fig2)

# Maintenance Action Logic
rul = engine_df.loc[engine_df.engine_id == selected_engine, "predicted_rul"].values[0]
if rul <= 15:
    st.error(f"Engine {selected_engine} should be replaced immediately! RUL: {rul}")
elif rul <= threshold:
    st.warning(f"Engine {selected_engine} needs inspection soon. RUL: {rul}")
else:
    st.success(f"Engine {selected_engine} is operating normally. RUL: {rul}")

# Optional: Upload for batch predictions
st.sidebar.header("Batch Prediction")
uploaded_file = st.sidebar.file_uploader("Upload Engine Data (CSV)", type=["csv"])
if uploaded_file:
    uploaded_df = pd.read_csv(uploaded_file)
    st.write("Uploaded Batch Data")
    st.dataframe(uploaded_df)
    

# Save to file
# file_path = "/mnt/data/streamlit_app.py"
# with open(file_path, "w") as f:
#    f.write(streamlit_template)

# file_path
