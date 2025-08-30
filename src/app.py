import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go
import plotly.express as px

from utils.plot import get_color, color_map
from pipeline import run_pipeline, datasets, load_datasets
from hybrid_model import HybridModel, OriginalModel
import math
import json


st.set_page_config(layout="wide")
st.title("🧠 Integration of Decision Explanation in Stock Trend Prediction")

# Side bar
st.sidebar.header("⚙️ Configuration")

## Select a file data
load_datasets()
file_options = ["-- Select dataset --"] + list(datasets.keys())
selected_file = st.sidebar.selectbox("📁 Select CSV file", file_options)

## Upload a file data
### - read data by pandas, check data have column 'close' or not...
uploaded_file = st.sidebar.file_uploader("📤 Or upload your own CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        user_df = pd.read_csv(uploaded_file)
        if 'close' not in user_df.columns and 'time' not in user_df.columns:
            st.error("❌ Uploaded file must contain a 'close' column and a 'time' column.")
            st.stop()
        else:
            st.session_state['user_df'] = user_df
            file_path = uploaded_file.name.split('.')[0].replace(' ', '_')
            file_path = f'upload/{file_path}_{len(user_df)}.csv'
            st.session_state['uploaded_file'] = file_path
            os.makedirs('upload', exist_ok=True)
            user_df.to_csv(file_path)
            st.success(f"✅ File uploaded: `{uploaded_file.name}` with {len(user_df)} rows.")
    except Exception as e:
        st.error(f"❌ Failed to read uploaded file: {e}")
        st.stop()

## Select a model
st.sidebar.markdown("---")
available_models = {
    "ELACB": [HybridModel, ['LSTMA', 'ANNR', 'TCN'], 'ANNC', 'CB', True],
    # "ETACB": [HybridModel, ['TCNA', 'ANNR', 'TCN'], 'ANNC', 'CB', True],
    # "ELARF": [HybridModel, ['LSTMA', 'ANNR', 'TCNA'], 'ANNC', 'RF', True],
    "Original Model (with EMD)": [OriginalModel, ['ANNR'], 'ANNC', 'RF', True],
    "Non-decomposing Model": [OriginalModel, ['ANNR'], 'ANNC', 'RF', False],
}
selected_model_name = st.sidebar.selectbox("🧠 Select Model", list(available_models.keys()))

# Main
## Check file csv
file_path = None
if 'uploaded_file' in st.session_state:
    file_path = st.session_state['uploaded_file']
    raw_df = st.session_state['user_df']
elif selected_file != "-- Select dataset --":
    file_path = datasets[selected_file]
    raw_df = pd.read_csv(file_path)
    st.success(f"✅ Loaded: `{selected_file}` with {len(raw_df)} rows")

if selected_model_name and file_path is not None:
    st.session_state['RAW_DF'] = raw_df
    raw_series = raw_df['close'].values
    time_series = raw_df['time'].values

    ## Load or train a model
    selected_model = available_models[selected_model_name]
    with st.spinner("🚀 Running model pipeline, please wait..."):
        model, DF, led, evaluation, trade_result = run_pipeline(
            selected_model[0],  # Model class
            file_path,          # File path
            selected_model[1],  # Input models
            selected_model[2],  # Fusion model
            selected_model[3],  # Output model
            selected_model[4],  # use_emd
        )
        st.session_state['model'] = model
        DF['time'] = time_series
        st.session_state['DF'] = DF
        st.session_state['led'] = led
        st.session_state['evaluation'] = evaluation
        st.session_state['trade_result'] = trade_result

        PRED_DF = DF.copy()
        PRED_DF.dropna(subset=['time', 'pred_value'], inplace=True)
        pred_time_series = PRED_DF['time'].values
        pred_series = PRED_DF['pred_value'].values

    st.success("✅ Pipeline completed!")

    ## plot fig
    st.subheader("📈 Raw vs Predicted Signal")
    fig_raw = go.Figure()
    fig_raw.add_trace(go.Scatter(y=raw_series, x=time_series, mode='lines', name='Raw Signal', line=dict(color='blue')))
    fig_raw.add_trace(go.Scatter(y=pred_series, x=pred_time_series, mode='lines', name='Predicted Signal', line=dict(color='red')))
    fig_raw.update_layout(
        showlegend=True,
        height=500,
        xaxis=dict(
            rangeslider=dict(visible=True),
            type="date"
        )
    )
    st.plotly_chart(fig_raw, use_container_width=True)
    # Tab
    raw_data_tab, decomposition_tab, leg_graph_tab, submodels_tab, hybrid_model_tab = st.tabs([
                                                                                                "📉 Input Time Series",
                                                                                                "📋 Signal Decomposition",
                                                                                                "🧠 Layered Explainability Graph",
                                                                                                "🔧 Submodels Training Results",
                                                                                                "🏆 Hybrid Model Evaluation"
                                                                                            ])
    
    with raw_data_tab:
        st.subheader("📉 Raw Time Series Data")
        raw_df = st.session_state['RAW_DF']
        st.dataframe(raw_df, height=400)

    with decomposition_tab:
        DF = st.session_state['DF']
        model = st.session_state['model']

        imf_cols = []
        for col in DF.columns:
            if col.startswith('imf_'):
                imf_cols.append(col)
        num_imfs = len(imf_cols)
        if num_imfs > 0:
            st.subheader("📋 Signal Decomposition into IMFs")
            metrics = model.imfs_metrics
            cols = metrics.columns
            for idx, row in metrics.iterrows():
                i = idx
                with st.expander(f"🔍 IMF_{i}", expanded=True):
                    col1, col2 = st.columns([1, 4])

                    with col1:
                        st.markdown(f"#### IMF_{i}")
                        if 'term' in cols:
                            st.markdown(f"""
                                        **📈 Term:** {row['term']}
                                        """)
                        st.markdown(f"""
                            **📡 Frequency:** {row['Average Frequency (Hz)']:.4f} Hz  
                            **🔁 Cycles:** {row['Number of Cycles']:.4f}  
                            **🔊 Amplitude:** {row['Average Amplitude']:.4f}
                        """)
                    
                    with col2:
                        imf = DF[f'imf_{i}'].values
                        fig_raw = go.Figure()
                        fig_raw.add_trace(go.Scatter(y=imf, mode='lines', name=f'IMF {i}', line=dict(color='blue')))
                        fig_raw.update_layout(
                            showlegend=True,
                            height=300
                        )
                        st.plotly_chart(fig_raw, use_container_width=True)

        else:
            st.info("⚠️ No decomposition data available.")
    
    with leg_graph_tab:
        led = st.session_state['led']
        st.subheader("🧠 Layered Explainability Graph (LEG)")
        with open("xai/leg_graph.html", "r", encoding="utf-8") as f:
            html_template = f.read()
        graph_data_json = json.dumps(led)
        html_with_data = html_template.replace("{{LED_DATA}}", graph_data_json)
        st.components.v1.html(html_with_data, height=600, scrolling=True)

    
    with submodels_tab:
        st.subheader("🔧 Submodels Training Results")
        evaluation = st.session_state['evaluation']
        DF = st.session_state['DF']
        regression_metric = ["MAE", "MSE", "RMSE", "MAPE", "R_squared"]
        classification_metric = ["Accuracy", "Precision", "Recall", "F1-score", "ROC AUC"]

        for model_name, metrics in evaluation.items():
            if " " not in model_name:
                continue 
            
            parts = model_name.split(" ")
            model_name = parts[0]
            true_col = parts[1]
            pred_col = parts[2]
            PRED_DF = DF.copy()
            PRED_DF = PRED_DF[['time', true_col, pred_col]]
            time_series = PRED_DF['time'].values
            series = PRED_DF[true_col].values
            PRED_DF.dropna(subset=['time', true_col, pred_col], inplace=True)
            pred_time_series = PRED_DF['time'].values
            pred_series = PRED_DF[pred_col].values

            true_col = true_col.replace("_", " ")
            st.markdown(f"### 🧠 {model_name} for {true_col} prediction")

            fig_raw = go.Figure()
            fig_raw.add_trace(go.Scatter(y=series, x=time_series, mode='lines', name='Raw Signal', line=dict(color='blue')))
            fig_raw.add_trace(go.Scatter(y=pred_series, x=pred_time_series, mode='lines', name='Predicted Signal', line=dict(color='red')))
            fig_raw.update_layout(
                showlegend=True,
                height=500,
                xaxis=dict(
                    rangeslider=dict(visible=True),
                    type="date"
                )
            )
            st.plotly_chart(fig_raw, use_container_width=True)

            values, labels, color_labels = [], [], []

            is_classification = any(metric in metrics for metric in classification_metric)
            metric_list = classification_metric if is_classification else regression_metric

            for name in metric_list:
                if name in metrics:
                    val = metrics[name]
                    icon = get_color(val, name)
                    values.append(val)
                    labels.append(f"{name}")
                    color_labels.append(icon)

            fig = px.bar(
                x=values,
                y=labels,
                orientation='h',
                text=[f"{v:.4f}" for v in values],
                color=color_labels,
                color_discrete_map=color_map,
                labels=None
            )

            fig.update_layout(
                height=400,
                xaxis_title=None,
                yaxis_title=None,
                showlegend=False,
                margin=dict(l=40, r=10, t=30, b=30)
            )

            st.plotly_chart(fig, use_container_width=True)

    with hybrid_model_tab:
        st.subheader("🏆 Hybrid Model Evaluation Metrics")
        evaluation = st.session_state['evaluation']
        trade_result = st.session_state['trade_result']
        classification_metric = ["Accuracy", "Precision", "Recall", "F1-score", "ROC AUC"]
        for model_name, metrics in evaluation.items():
            if " " in model_name:
                continue 

            st.markdown(f"### 🧠 {model_name}")

            metric_list = classification_metric
            data = []

            for name in metric_list:
                if name in metrics:
                    val = metrics[name]
                    icon = get_color(val, name)
                    data.append({"Metric": name, "Value": val, "Color": icon})

            df = pd.DataFrame(data)

            fig = px.bar(
                df,
                x="Value",
                y="Metric",
                orientation='h',
                text=df["Value"].apply(lambda v: f"{v:.4f}"),
                color="Color",
                color_discrete_map=color_map,
            )

            fig.update_layout(
                height=400,
                xaxis_title=None,
                yaxis_title=None,
                showlegend=False,
                margin=dict(l=40, r=10, t=30, b=30)
            )

            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("## 📈 Trading Results")

        markdown = "| Metric | Value |\n|---|---|\n"
        for key, val in trade_result.items():
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                val_display = "NaN" if math.isnan(val) else "∞"
            else:
                val_display = f"{val:.2f}" if isinstance(val, float) else val
            markdown += f"| {key} | {val_display} |\n"

        st.markdown(markdown)


else:
    st.warning("⚠️ Please select a file or upload your own CSV file")