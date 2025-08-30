import numpy as np
import pandas as pd
import os
import csv

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
    
def save_trading_result(metrics, model_name, data_name, metric_file="saved/evaluation/trade.csv"):
    model_first_headers = ['Model', 'Dataset'] + list(metrics.keys())
    model_first_row = [model_name, data_name] + list(metrics.values())

    model_file_exists = os.path.isfile(metric_file)

    try:
        with open(metric_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if not model_file_exists or os.path.getsize(metric_file) == 0:
                writer.writerow(model_first_headers)
            writer.writerow(model_first_row)
        print(f"Đã cập nhật file: {metric_file}")
    except IOError as e:
        print(f"Lỗi khi ghi file {metric_file}: {e}")
    except Exception as e:
        print(f"Lỗi không xác định khi xử lý {metric_file}: {e}")

def evaluate_regression_models(DF, 
                                y_true_column_name, 
                                y_pred_column_name, 
                                model_name,
                                folder_data,
                                save_path='saved/evaluation/regressor.csv'):
    results = []

    DF = DF.copy()
    DF.dropna(subset=[y_true_column_name, y_pred_column_name], inplace=True)

    y_true = DF[y_true_column_name].values
    y_pred = DF[y_pred_column_name].values
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    results.append({
        'Model name': model_name,
        'Data': folder_data,
        'Column': y_true_column_name,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'R_squared': r2
    })

    df_result = pd.DataFrame(results)
    write_header = not os.path.exists(save_path)
    df_result.to_csv(save_path, index=False, mode='a', header=write_header)
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'R_squared': r2
    }

def evaluate_classifier_models(DF, 
                               y_true_column_name, 
                               y_prob_column_name, 
                               model_name,
                               folder_data,
                               save_path='saved/evaluation/classifier.csv'):
    results = []

    DF = DF.copy()
    DF.dropna(subset=[y_true_column_name, y_prob_column_name], inplace=True)

    valid_mask = ~(DF[y_true_column_name].isna() | DF[y_prob_column_name].isna())
    y_true = DF.loc[valid_mask, y_true_column_name].values
    y_prob = DF.loc[valid_mask, y_prob_column_name].values
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)

    results.append({
        'Model name': model_name,
        'Data': folder_data,
        'Column': y_true_column_name,
        "Accuracy": round(acc, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1-score": round(f1, 4),
        "ROC AUC": round(roc_auc, 4),
        "TP": cm[1, 1],
        "TN": cm[0, 0],
        "FP": cm[0, 1],
        "FN": cm[1, 0]
    })

    df_result = pd.DataFrame(results)
    write_header = not os.path.exists(save_path)
    df_result.to_csv(save_path, index=False, mode='a', header=write_header)
    return {
        "Accuracy": round(acc, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1-score": round(f1, 4),
        "ROC AUC": round(roc_auc, 4),
        "TP": cm[1, 1],
        "TN": cm[0, 0],
        "FP": cm[0, 1],
        "FN": cm[1, 0]
    }

def evaluate_ensemble_models(DF,
                             y_true_column_name, 
                             y_pred_column_name, 
                             model_name,
                             folder_data,
                             save_path='saved/evaluation/ensembler.csv'):
    results = []

    DF = DF.copy()
    DF.dropna(subset=[y_true_column_name, y_pred_column_name], inplace=True)

    y_true = DF[y_true_column_name].values
    y_pred = DF[y_pred_column_name].values

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    results.append({
        'Model name': model_name,
        'Data': folder_data,
        "Accuracy": round(acc, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1-score": round(f1, 4),
        "ROC AUC": round(roc_auc, 4),
        "TP": cm[1, 1],
        "TN": cm[0, 0],
        "FP": cm[0, 1],
        "FN": cm[1, 0]
    })

    df_result = pd.DataFrame(results)
    write_header = not os.path.exists(save_path)
    df_result.to_csv(save_path, index=False, mode='a', header=write_header)

    return {
        "Accuracy": round(acc, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1-score": round(f1, 4),
        "ROC AUC": round(roc_auc, 4),
        "TP": cm[1, 1],
        "TN": cm[0, 0],
        "FP": cm[0, 1],
        "FN": cm[1, 0]
    }