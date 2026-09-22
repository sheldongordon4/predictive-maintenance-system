import io

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pipeline

# Must be the first Streamlit command in the script, before anything else
# that could emit a Streamlit UI element -- including get_artifacts() below,
# whose @st.cache_resource(show_spinner=...) shows a spinner (itself a
# Streamlit command) on a cache miss, which would otherwise become the de
# facto "first command" and make this call raise
# StreamlitSetPageConfigMustBeFirstCommandError.
st.set_page_config(page_title="Engine Health Dashboard", layout="wide")

# Set Pandas display options (updated to support enough elements)
pd.set_option("styler.render.max_elements", 600000)


@st.cache_resource(show_spinner="Loading model...")
def get_artifacts():
    """Load the fitted preprocessor (imputer + scaler) and the trained
    LightGBM model once per session, rather than on every widget interaction.
    Both come from engine_model_dev.ipynb and must be used together -- the
    model was trained on data scaled by this exact preprocessor.
    """
    return pipeline.load_artifacts()


@st.cache_data(show_spinner="Running RUL predictions...")
def compute_predictions(file_bytes):
    """Parse an uploaded CSV and run the full inference pipeline.

    Cached on the raw file bytes, so this only re-runs when a genuinely new
    file is uploaded -- not on every slider/selectbox interaction.
    """
    df = pd.read_csv(io.BytesIO(file_bytes))
    pipeline.validate_schema(df)
    preprocessor, model = get_artifacts()
    df = df.copy()
    df['predicted_rul'] = pipeline.predict_rul(df, preprocessor, model)
    return df


# Fail fast on startup if the model artifacts aren't available, before the
# user even uploads a file.
try:
    get_artifacts()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# Title and description
st.title("Engine Health Monitoring Dashboard")
st.markdown("Monitor, assess, and predict the health of turbofan engines using NASA CMAPSS data.")

# Sidebar inputs
st.sidebar.header("Configuration")
threshold = st.sidebar.slider("RUL Threshold (for Inspection)", 10, 70, 30)
uploaded_file = st.sidebar.file_uploader("Upload Engine Sensor CSV", type=["csv"])

# Main logic
if uploaded_file is not None:
    try:
        df_predict = compute_predictions(uploaded_file.getvalue())
    except ValueError as e:
        # Covers essentially every "something's wrong with this data" case:
        # pandas' own CSV-parse errors (ParserError, EmptyDataError,
        # UnicodeDecodeError) and sklearn's dtype/NaN/inf validation errors
        # are all ValueError subclasses, as is pipeline.validate_schema's
        # missing-column error -- so `e` already carries a specific,
        # actionable message in every one of those cases.
        st.error(str(e))
        st.stop()
    except Exception as e:
        # Anything else (e.g. a LightGBM-internal error, which is NOT a
        # ValueError) is a genuine surprise, not a data problem -- don't
        # guess at a cause we don't actually know.
        st.error(f"Unexpected error while processing the uploaded file: {e}")
        st.stop()

    # `df` and `df_predict` are the same underlying data (raw sensor columns
    # plus predicted_rul); kept as two names since `df` is used below for raw
    # sensor trends and `df_predict` gains an 'action' column derived from
    # the (frequently-changing) threshold slider. st.cache_data returns a
    # fresh copy each call, so mutating df_predict below never touches the cache.
    df = df_predict

    # Preview dataframe
    st.write("Uploaded Data Preview:")
    st.dataframe(df.head())

    # Extract and save engine IDs
    engine_ids = df['engine_id'].unique()

    sensor_columns = [col for col in pipeline.MODEL_FEATURES if col != 'cycle_time']
    selected_sensor = st.sidebar.selectbox("Sensor to Monitor", sensor_columns)

    # Categorize engine health status. Kept outside compute_predictions since
    # it depends on `threshold`, which changes on every slider interaction --
    # re-deriving this cheap column is far cheaper than re-running predictions.
    df_predict['action'] = df_predict['predicted_rul'].apply(
        lambda x: 'Repair' if x <= 15 else ('Inspect' if x <= threshold else 'Monitor')
    )

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
        subset_df.style.map(highlight_action, subset=["action"]),
        use_container_width=True
    )

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
    plt.close(fig1)

    # Engine-specific diagnostics
    selected_engine = st.selectbox("Drill-down: Select Engine ID", engine_ids)

    # Filter and plot selected engine's sensor trend
    engine_df = df[df.engine_id == selected_engine].sort_values("cycle_time")
    if selected_sensor in sensor_columns:
        sensor_data = engine_df[["cycle_time", selected_sensor]]

        st.subheader(f"{selected_sensor} Over Time for Engine {selected_engine}")
        if len(sensor_data) < 2:
            # A line plot with 0-1 points renders as a blank chart with no
            # visible explanation -- e.g. this always happens for
            # test_data.csv, which has exactly one row (cycle) per engine.
            st.info(
                f"Engine {selected_engine} has only {len(sensor_data)} recorded "
                "cycle(s) in this file, so there's no trend to plot. Upload a "
                "file with multiple rows per engine (e.g. training_data.csv, "
                "or the full test_data.txt trajectories) to see sensor trends "
                "over time."
            )
        else:
            fig2, ax2 = plt.subplots()
            sns.lineplot(data=sensor_data, x="cycle_time", y=selected_sensor, marker="o", ax=ax2)
            st.pyplot(fig2)
            plt.close(fig2)
    else:
        st.warning(f"{selected_sensor} not found for Engine {selected_engine}")

    # Maintenance recommendation
    latest_row = df_latest[df_latest['engine_id'] == selected_engine]
    if latest_row.empty:
        st.warning(f"No data available for Engine {selected_engine}.")
    else:
        latest_rul = latest_row['predicted_rul'].values[0]

        if latest_rul <= 15:
            st.error(f"Engine {selected_engine} should be repaired immediately! RUL: {latest_rul}")
        elif latest_rul <= threshold:
            st.warning(f"Engine {selected_engine} needs inspection soon. RUL: {latest_rul}")
        else:
            st.success(f"Engine {selected_engine} is operating normally. RUL: {latest_rul}")

else:
    st.info("Please upload a CSV file to begin analysis.")
