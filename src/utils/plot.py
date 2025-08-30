import plotly.graph_objects as go

def plot_imfs_interactive(imfs, raw_signal=None):
    figs = []
    num_imfs = imfs.shape[0]

    if raw_signal is not None:
        fig_raw = go.Figure()
        fig_raw.add_trace(go.Scatter(y=raw_signal, mode='lines', name='Raw Signal'))
        fig_raw.update_layout(
            showlegend=True,
            height=500,
            xaxis=dict(
                rangeslider=dict(visible=True),
                type="linear"
            )
        )
        figs.append(('Raw Signal', fig_raw))

    for i in range(num_imfs):
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=imfs[i], mode='lines', name=f"IMF {i}"))
        fig.update_layout(
            showlegend=False,
            height=350
        )
        figs.append((f"IMF {i}", fig))

    return figs

def get_color(value, metric_name):
    thresholds = {
        "MAE": (0.02, 0.05),
        "MSE": (0.001, 0.005),
        "RMSE": (0.03, 0.07),
        "MAPE": (0.05, 0.10),
        "R_squared": (0.80, 0.90),
        "Accuracy": (0.75, 0.85),
        "Precision": (0.70, 0.85),
        "Recall": (0.75, 0.85),
        "F1-score": (0.75, 0.85),
        "ROC AUC": (0.80, 0.90)
    }

    low, high = thresholds[metric_name]
    if metric_name in ["R_squared", "Accuracy", "Precision", "Recall", "F1-score", "ROC AUC"]:
        if value >= high:
            return "🟢"
        elif value >= low:
            return "🟡"
        else:
            return "🔴"
    else:
        if value <= low:
            return "🟢"
        elif value <= high:
            return "🟡"
        else:
            return "🔴"

color_map = {
    "🟢": "#00cc96",
    "🟡": "#ffe70e",
    "🔴": "#ff4b4b"
}