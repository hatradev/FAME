import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def ranking_regressors(file_path, column='Model name', selected_models=['TCN', 'TCN_ATT', 'ANNR', 'LSTM_ATT', 'LSTM', 'GRU']):
    df = pd.read_csv(file_path)

    if selected_models is not None:
        df = df[df[column].isin(selected_models)]
    
    grouped = df.groupby(['Column', column]).agg({
        'MAE': 'mean',
        'MSE': 'mean',
        'RMSE': 'mean',
        'MAPE': 'mean',
        'R_squared': 'mean'
    }).reset_index()

    sorted_grouped = grouped.sort_values(
        by=['Column', 'R_squared', 'RMSE', 'MAE'],
        ascending=[True, False, True, True]
    )

    rank_dict = {
        col: sorted_grouped[sorted_grouped['Column'] == col][column].tolist()
        for col in sorted_grouped['Column'].unique()
    }

    return rank_dict, sorted_grouped

def ranking_classifiers(file_path, column='Model name', selected_models=None):
    df = pd.read_csv(file_path)

    if selected_models:
        df = df[df[column].isin(selected_models)]
    
    grouped = df.groupby(column).mean(numeric_only=True).reset_index()

    sorted_grouped = grouped.sort_values(
        by=['F1-score', 'Accuracy', 'ROC AUC'],
        ascending=[False, False, False]
    )

    return sorted_grouped

def ranking_models(file_path, column='Model name', selected=None,
                   mean_columns=['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC AUC'],
                   sum_columns=[],
                   sorted_columns=['F1-score', 'Accuracy'], 
                   ascending=[False, False]):
    df = pd.read_csv(file_path)
    if selected:
        df = df[df[column].isin(selected)]

    mean_df = df.groupby(column)[mean_columns].mean()
    sum_df = df.groupby(column)[sum_columns].sum()
    final_df = pd.concat([mean_df, sum_df], axis=1).reset_index()

    if sorted_columns and ascending:
        sorted_df = final_df.sort_values(by=sorted_columns, ascending=ascending)

    model_rank = sorted_df[column].tolist()

    return sorted_df, model_rank

def draw_heatmap_regressors(sorted_grouped, save_dir):
    df = sorted_grouped.copy()

    df['Model (Term)'] = df['Model name'] + " (" + df['Column'].str.replace('_', ' ') + ")"
    df.set_index('Model (Term)', inplace=True)

    metrics = ['MAE', 'RMSE', 'MAPE', 'R_squared']
    df_metrics = df[metrics]

    df_norm = df_metrics.copy()
    for col in df_norm.columns:
        if col == 'R_squared':
            df_norm[col] = (df_metrics[col] - df_metrics[col].min()) / (df_metrics[col].max() - df_metrics[col].min())
        else:
            df_norm[col] = 1 - (df_metrics[col] - df_metrics[col].min()) / (df_metrics[col].max() - df_metrics[col].min())

    annot_matrix = df_norm.copy()
    for row in df_norm.index:
        for col in df_norm.columns:
            original = df_metrics.loc[row, col]
            normalized = df_norm.loc[row, col]
            annot_matrix.loc[row, col] = f"{original:.4f} ({normalized:.2f})"

    n_rows = df_norm.shape[0]
    n_cols = df_norm.shape[1]
    cell_height = 0.4
    cell_width = 2.2
    fig_height = max(6, n_rows * cell_height)
    fig_width = max(8, n_cols * cell_width)

    plt.figure(figsize=(fig_width, fig_height))
    sns.heatmap(df_norm.astype(float), cmap="YlGnBu", annot=annot_matrix, fmt="", linewidths=0.5,
                cbar_kws={'label': 'Normalized Score'})
    plt.title("Regression Model Performance Heatmap (Actual + Normalized Values)", fontsize=14, pad=14)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir)
    plt.show()



