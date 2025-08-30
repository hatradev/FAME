import pandas as pd
import numpy as np
from PyEMD import EMD
from utils.preprocess import determine_direction, determine_difference,  return_data_with_lagged_arrays, label_imf, calculate_imfs_metrics, split_dataset, add_partial_column
from utils.runner import write_model, build_explainer
from eval.metric import *
from sklearn.preprocessing import MinMaxScaler
import shap
from lime import lime_tabular
import json
import joblib
from tensorflow.keras.saving import load_model
from model.custom_object import custom_objects
from backtesting import simulate_trading
from xai.main import determine_reliable_predictions

SAVED_DIR = 'saved'
LAG = 14

class HybridModel:
    def __init__(self, name, dataset, decomposition=EMD()):
        self.name = name
        self.parts = name.split("_")
        self.dataset = dataset
        self.decomposition = decomposition
        self.models = {}
        self.explainers = {}
        self.model_dirs = {}
        self.explaination_dirs = {}
        self.root_folder = f'{SAVED_DIR}/{self.name}/{self.dataset}'

        os.makedirs(f'{self.root_folder}/preprocess', exist_ok=True)

    def load(self, df_path, explaination_path, model_path):
        self.DF = pd.read_csv(df_path)

        with open(explaination_path, 'r') as f:
            self.explaination_dirs = json.load(f)

        with open(model_path, 'r') as f:
            self.model_dirs = json.load(f)
        
        for key, path in self.model_dirs.items():
            if path.endswith('model'):
                self.models[key] = joblib.load(path)
            elif path.endswith('model.keras'):
                self.models[key] = load_model(path, custom_objects=custom_objects)
            elif path.endswith('imfs_metrics.csv'):
                self.imfs_metrics = pd.read_csv(path)


    def pretrain_process(self, close_price, scaler=MinMaxScaler(), lag=LAG):
        # read and scale data
        self.DF = pd.DataFrame()
        price_org = close_price.flatten()
        self.DF['price_org'] = price_org

        scaled = scaler.fit_transform(close_price)
        self.DF['price_scaled'] = scaled.flatten()
        self.DF['dir_obs_price'] = determine_direction(close_price)

        # decompose data
        imfs = self.decomposition(self.DF['price_scaled'].values)
        num_imfs = imfs.shape[0]
        for i in range(num_imfs):
            self.DF["imf_"+str(i)] = imfs[i]

        dataset = {}
        # generate classifier dataset
        self.DF['dir_obs_imf_0'] = determine_direction(self.DF['imf_0'])
        lagged_arrays = return_data_with_lagged_arrays(self.DF['imf_0'].values, lag)
        x_classifier = lagged_arrays[:, 1:(lagged_arrays.shape[1])]
        y_classifier = self.DF['dir_obs_imf_0'].values[lag:]
        dataset['classifier'] = (x_classifier, y_classifier)

        # group data by frequency via KMean for Regressor Dataset
        self.imfs_metrics = calculate_imfs_metrics(imfs)
        range_term = label_imf(self.imfs_metrics)
        term_col = [''] * num_imfs
        for term, val in range_term.items():
            self.DF[term] = imfs[val].sum(axis=0)
            lagged_arrays = return_data_with_lagged_arrays(self.DF[term].values, lag)
            x_regressor = lagged_arrays[:, 1:(lagged_arrays.shape[1])]
            y_regressor = lagged_arrays[0:lagged_arrays.shape[0], 0]
            dataset[f'{term}_regressor'] = (x_regressor, y_regressor)

            for idx in val:
                term_col[idx] = term
        self.imfs_metrics['term'] = term_col
        self.imfs_metrics.to_csv(f'{self.root_folder}/preprocess/imfs_metrics.csv')
        self.model_dirs["imfs_metrics"] = f'{self.root_folder}/preprocess/imfs_metrics.csv'
        
        # Ensembler 
        dataset['ensembler'] = self.DF['dir_obs_price'].values[lag:]
        return dataset, scaler
        
    def fit(self, close_price, regressor_drivers, classifier_driver, ensembler_driver, scaler=MinMaxScaler(), lag=LAG):
        self.scaler = scaler
        self.lag = lag
        self.regressor_drivers = regressor_drivers
        self.classifier_driver = classifier_driver
        self.ensembler_driver = ensembler_driver

        dataset, self.scaler = self.pretrain_process(close_price, self.scaler, self.lag)

        # Train predictors - 1st layer
        keys = ['short_term_regressor', 'mid_term_regressor', 'long_term_regressor', 'classifier']
        predictor_drivers = regressor_drivers + [classifier_driver]
        output_columns_1 = ['short_score', 'mid_score', 'long_score', 'pred_imf_0_class']
        input_columns_2 = ['diff_short_score', 'diff_mid_score', 'diff_long_score', 'pred_imf_0_class']
        VALID_DF = pd.DataFrame()
        TEST_DF = pd.DataFrame()

        for key, (predictor_driver, reshape_input), output_column_1, input_column_2 in zip(keys, predictor_drivers, output_columns_1, input_columns_2):
            ## Split dataset anf reshape
            x, y = dataset[key]
            x_train, x_valid, x_test = split_dataset(x)
            y_train, y_valid, y_test = split_dataset(y)

            if reshape_input:
                x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
                x_valid = x_valid.reshape((x_valid.shape[0], x_valid.shape[1], 1))
                x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

            ## Train and save model
            model, path = write_model(x_train, y_train, predictor_driver, f'{self.root_folder}/{key}/')
            self.models[key] = model
            self.model_dirs[key] = path

            ## Predict
            y_valid_pred = model.predict(x_valid)
            y_valid_pred = y_valid_pred.reshape(-1)

            y_test_pred = model.predict(x_test)
            y_test_pred = y_test_pred.reshape(-1)

            ## Save to DF
            VALID_DF[output_column_1] = np.reshape(y_valid_pred, (len(y_valid_pred),))
            TEST_DF[output_column_1] = np.reshape(y_test_pred, (len(y_test_pred),))

            if input_column_2 == f'diff_{output_column_1}':
                VALID_DF[input_column_2] = determine_difference(VALID_DF[output_column_1].values)
                TEST_DF[input_column_2] = determine_difference(TEST_DF[output_column_1].values)

            y_pred = np.concatenate([y_valid_pred.flatten(), y_test_pred.flatten()])
            self.DF = add_partial_column(self.DF, output_column_1, y_pred, lag+y_train.shape[0])

            ## Train explainer and save
            explain_file_path = f'{self.root_folder}/{key}/explaination.csv'
            explainer = build_explainer(model, x_train, x_test, explain_file_path, reshape_input)

            self.explainers[key] = explainer
            self.explaination_dirs[input_column_2] = explain_file_path
        
        ## Calculate prediction
        self.DF['scaled_pred'] = self.DF[['short_score', 'mid_score', 'long_score']].sum(axis=1, min_count=1)

        scaled_values = self.DF['scaled_pred'].values.reshape(-1, 1)
        mask = ~np.isnan(scaled_values).flatten()
        original_values = np.full_like(scaled_values, np.nan)
        original_values[mask] = self.scaler.inverse_transform(scaled_values[mask])
        self.DF['pred_value'] = original_values.flatten()

        # Train ensembler - 2nd layer
        key = 'ensembler'

        ## get dataset
        y_tmp, y_train, y_test = split_dataset(dataset[key])
        idx = lag + y_tmp.shape[0] + y_train.shape[0] + 1
        VALID_DF['target'] = y_train
        TEST_DF['target'] = y_test

        VALID_DF.dropna(subset=input_columns_2 + ['target'], inplace=True)
        TEST_DF.dropna(subset=input_columns_2 + ['target'], inplace=True)

        x_train = VALID_DF[input_columns_2].values
        y_train = VALID_DF['target'].values
        x_test = TEST_DF[input_columns_2].values

        ## Train and save model
        model, path = write_model(x_train, y_train, ensembler_driver, f'{self.root_folder}/{key}/')
        self.models[key] = model
        self.model_dirs[key] = path

        ## Predict
        final_prob = model.predict_proba(x_test)
        final_prob = final_prob[:, 1]
        final_pred = np.where(final_prob > 0.5, 1, 0)

        ## Save to DF
        self.DF = add_partial_column(self.DF, 'final_prob', final_prob, idx)
        self.DF = add_partial_column(self.DF, 'final_pred', final_pred, idx)

        ## Train explainer and save
        explainer = shap.TreeExplainer(model=model, feature_names=input_columns_2)
        shap_values = explainer(x_test)
        shap.initjs()

        shap_df = pd.DataFrame(shap_values.values, columns=shap_values.feature_names)
        explain_file_path = f'{self.root_folder}/{key}/explaination.csv'
        shap_df.to_csv(explain_file_path, index=False)
        
        self.explainers[key] = explainer
        self.explaination_dirs['final_pred'] = explain_file_path

        with open(f'{self.root_folder}/model.json', 'w') as f:
            json.dump(self.model_dirs, f, indent=4)
        
        with open(f'{self.root_folder}/explaination.json', 'w') as f:
            json.dump(self.explaination_dirs, f, indent=4)
        
        self.DF.to_csv(f'{self.root_folder}/result.csv')
    
    def trade(self):
        DF = self.DF.copy()
        DF = DF[['price_org', 'final_prob', 'dir_obs_price']]
        DF.dropna(subset=['price_org', 'final_prob', 'dir_obs_price'], inplace=True)
        y_prob_1 = DF['final_prob'].values
        y_prob_0 = 1 - y_prob_1
        y_prob = np.column_stack((y_prob_0, y_prob_1))
        y_true = DF['dir_obs_price'].values

        final = determine_reliable_predictions(y_prob, y_true)
        final = ['up' if x == 1 else 'down' for x in final]
        trades_df, metrics = simulate_trading(DF['price_org'].values, final, initial_balance=1000.0, trade_percent=0.1, sl_pct=0.015, tp_pct=0.02)
        return trades_df, metrics
    
    def evaluate(self):
        os.makedirs('saved/evaluation', exist_ok=True)
        r_preds = ['short_score', 'mid_score', 'long_score']
        r_trues = ['short_term', 'mid_term', 'long_term']

        c_pred = 'pred_imf_0_class'
        c_true = 'dir_obs_imf_0'

        e_pred = 'final_pred'
        e_true = 'dir_obs_price'

        self.evaluation = {}
        for r_pred, r_true, name  in zip(r_preds, r_trues, self.parts[1:4]):
            self.evaluation[f'{name} {r_true} {r_pred}'] = evaluate_regression_models(self.DF, r_true, r_pred, name, self.dataset)
        self.evaluation[f'{self.parts[4]} {c_true} {c_pred}'] = evaluate_classifier_models(self.DF, c_true, c_pred, self.parts[4], self.dataset)
        self.evaluation[self.name] = evaluate_ensemble_models(self.DF, e_true, e_pred, self.name, self.dataset)

        return self.evaluation
    
class OriginalModel:
    def __init__(self, name, dataset, decomposition=EMD()):
        self.name = name
        self.parts = name.split("_")
        self.dataset = dataset
        self.decomposition = decomposition
        self.root_folder = f'{SAVED_DIR}/{self.name}/{self.dataset}'
        os.makedirs(f'{self.root_folder}/preprocess', exist_ok=True)
        self.models = {}
        self.model_dirs = {}
        self.explaination_dirs = {}
        self.explainers = {}

    def load(self, df_path, explaination_path, model_path):
        self.DF = pd.read_csv(df_path)

        with open(explaination_path, 'r') as f:
            self.explaination_dirs = json.load(f)

        with open(model_path, 'r') as f:
            self.model_dirs = json.load(f)
        
        self.num_imfs = -1
        for key, path in self.model_dirs.items():
            if path.endswith('model'):
                self.models[key] = joblib.load(path)
            elif path.endswith('model.keras'):
                self.models[key] = load_model(path, custom_objects=custom_objects)
                self.num_imfs += 1
            elif path.endswith('imfs_metrics.csv'):
                self.imfs_metrics = pd.read_csv(path)

    def pretrain_process(self, close_price, scaler=MinMaxScaler(), lag=LAG):
        # read and scale data
        self.DF = pd.DataFrame()
        price_org = close_price.flatten()
        self.DF['price_org'] = price_org

        scaled = scaler.fit_transform(close_price)
        self.DF['price_scaled'] = scaled.flatten()
        self.DF['dir_obs_price'] = determine_direction(close_price)

        dataset = {}
        # decompose data
        if self.decomposition:
            imfs = self.decomposition(self.DF['price_scaled'].values)
            self.num_imfs = imfs.shape[0]
            self.imfs_metrics = calculate_imfs_metrics(imfs)
            self.imfs_metrics.to_csv(f'{self.root_folder}/preprocess/imfs_metrics.csv')
            self.model_dirs["imfs_metrics"] = f'{self.root_folder}/preprocess/imfs_metrics.csv'
            for i in range(self.num_imfs):
                self.DF["imf_"+str(i)] = imfs[i]
                lagged_arrays = return_data_with_lagged_arrays(self.DF["imf_"+str(i)].values, lag)
                x_regressor = lagged_arrays[:, 1:(lagged_arrays.shape[1])]
                y_regressor = lagged_arrays[0:lagged_arrays.shape[0], 0]
                dataset[f'{i}_regressor'] = (x_regressor, y_regressor)

            # generate classifier dataset
            self.DF['dir_obs_imf_0'] = determine_direction(self.DF['imf_0'])
            lagged_arrays = return_data_with_lagged_arrays(self.DF['imf_0'].values, lag)
            x_classifier = lagged_arrays[:, 1:(lagged_arrays.shape[1])]
            y_classifier = self.DF['dir_obs_imf_0'].values[lag:]
            dataset['classifier'] = (x_classifier, y_classifier)
        else:
            lagged_arrays = return_data_with_lagged_arrays(self.DF['price_scaled'].values, lag)
            x = lagged_arrays[:, 1:(lagged_arrays.shape[1])]
            y_regressor = lagged_arrays[0:lagged_arrays.shape[0], 0]
            dataset['regressor'] = (x, y_regressor)

            # generate classifier dataset
            y_classifier = self.DF['dir_obs_price'].values[lag:]
            dataset['classifier'] = (x, y_classifier)
            
        # Ensembler 
        dataset['ensembler'] = self.DF['dir_obs_price'].values[lag:]
        return dataset, scaler
        
    def fit(self, close_price, regressor_drivers, classifier_driver, ensembler_driver, scaler=MinMaxScaler(), lag=LAG):
        self.scaler = scaler
        self.lag = lag
        self.regressor_drivers = regressor_drivers
        self.classifier_driver = classifier_driver
        self.ensembler_driver = ensembler_driver

        dataset, self.scaler = self.pretrain_process(close_price, self.scaler, self.lag)

        # Train predictors - 1st layer
        if self.decomposition is None:
            keys = ['regressor', 'classifier']
            predictor_drivers = regressor_drivers + [classifier_driver]
            output_columns_1 = ['pred_regre', 'pred_class']
            input_columns_2 = ['diff_pred_regre', 'pred_class']
        else:
            keys = [f'{i}_regressor' for i in range(self.num_imfs)] + ['classifier']
            predictor_drivers = regressor_drivers * (self.num_imfs) + [classifier_driver]
            output_columns_1 = [f'pred_imf_{i}' for i in range(self.num_imfs)] + ['pred_imf_0_class']
            input_columns_2 = [f'diff_pred_imf_{i}' for i in range(self.num_imfs)] + ['pred_imf_0_class']

        VALID_DF = pd.DataFrame()
        TEST_DF = pd.DataFrame()

        for key, (predictor_driver, reshape_input), output_column_1, input_column_2 in zip(keys, predictor_drivers, output_columns_1, input_columns_2):
            ## Split dataset anf reshape
            x, y = dataset[key]
            x_train, x_valid, x_test = split_dataset(x)
            y_train, y_valid, y_test = split_dataset(y)

            if reshape_input:
                x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
                x_valid = x_valid.reshape((x_valid.shape[0], x_valid.shape[1], 1))
                x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

            ## Train and save model
            model, path = write_model(x_train, y_train, predictor_driver, f'{self.root_folder}/{key}/')
            self.models[key] = model
            self.model_dirs[key] = path

            ## Predict
            y_valid_pred = model.predict(x_valid)
            y_valid_pred = y_valid_pred.reshape(-1)

            y_test_pred = model.predict(x_test)
            y_test_pred = y_test_pred.reshape(-1)

            ## Save to DF
            VALID_DF[output_column_1] = np.reshape(y_valid_pred, (len(y_valid_pred),))
            TEST_DF[output_column_1] = np.reshape(y_test_pred, (len(y_test_pred),))

            if input_column_2 == f'diff_{output_column_1}':
                VALID_DF[input_column_2] = determine_difference(VALID_DF[output_column_1].values)
                TEST_DF[input_column_2] = determine_difference(TEST_DF[output_column_1].values)

            y_pred = np.concatenate([y_valid_pred.flatten(), y_test_pred.flatten()])
            self.DF = add_partial_column(self.DF, output_column_1, y_pred, lag+y_train.shape[0])
        
        ## Calculate prediction
        self.DF['scaled_pred'] = self.DF[output_columns_1[:-1]].sum(axis=1, min_count=1)

        scaled_values = self.DF['scaled_pred'].values.reshape(-1, 1)
        mask = ~np.isnan(scaled_values).flatten()
        original_values = np.full_like(scaled_values, np.nan)
        original_values[mask] = self.scaler.inverse_transform(scaled_values[mask])
        self.DF['pred_value'] = original_values.flatten()

        # Train ensembler - 2nd layer
        key = 'ensembler'

        ## get dataset
        VALID_DF['pred_sum'] = VALID_DF[input_columns_2].sum(axis=1)
        VALID_DF['dir_pred_sum'] = determine_direction(VALID_DF['pred_sum'].values)
        VALID_DF['diff_pred_sum'] = determine_difference(VALID_DF['pred_sum'].values)

        TEST_DF['pred_sum'] = TEST_DF[input_columns_2].sum(axis=1)
        TEST_DF['dir_pred_sum'] = determine_direction(TEST_DF['pred_sum'].values)
        TEST_DF['diff_pred_sum'] = determine_difference(TEST_DF['pred_sum'].values)

        y_tmp, y_train, y_test = split_dataset(dataset[key])
        idx = lag + y_tmp.shape[0] + y_train.shape[0] + 1
        VALID_DF['target'] = y_train
        TEST_DF['target'] = y_test

        VALID_DF.dropna(subset=["dir_pred_sum", "diff_pred_sum", "target"], inplace=True)
        TEST_DF.dropna(subset=["dir_pred_sum", "diff_pred_sum", "target"], inplace=True)

        x_train = VALID_DF[["dir_pred_sum", "diff_pred_sum"]].values
        y_train = VALID_DF['target'].values
        x_test = TEST_DF[["dir_pred_sum", "diff_pred_sum"]].values

        ## Train and save model
        model, path = write_model(x_train, y_train, ensembler_driver, f'{self.root_folder}/{key}/')
        self.models[key] = model
        self.model_dirs[key] = path

        ## Predict
        final_prob = model.predict_proba(x_test)
        final_prob = final_prob[:, 1]
        final_pred = np.where(final_prob > 0.5, 1, 0)

        ## Save to DF
        self.DF = add_partial_column(self.DF, 'final_prob', final_prob, idx)
        self.DF = add_partial_column(self.DF, 'final_pred', final_pred, idx)

        ## Train explainer and save
        feature_names = ["dir_pred_sum", "diff_pred_sum"]

        explainer = lime_tabular.LimeTabularExplainer(x_train, mode="classification", feature_names=feature_names, class_names=["Down","Up"])  
        explanation_predict_proba = []
        explanation_dicts = []
        for i in range(x_test.shape[0]):
            explanation = explainer.explain_instance(x_test[i],model.predict_proba, num_features=x_train.shape[1])
            explanation_predict_proba.append(explanation.predict_proba)
            exp_dict = dict(explanation.as_list())
            explanation_dicts.append(exp_dict)

        explain_df = pd.DataFrame(explanation_dicts)
        # explain_df.insert(0, 'index', range(x_test.shape[0]))
        # probas = pd.DataFrame(explanation_predict_proba, columns=["proba_Down", "proba_Up"])
        # explain_df = pd.concat([probas, explain_df], axis=1)
        explain_file_path = f'{self.root_folder}/{key}/explaination.csv'
        explain_df.to_csv(explain_file_path, index=False)
        
        self.explainers[key] = explainer
        self.explaination_dirs['final_pred'] = explain_file_path

        with open(f'{self.root_folder}/model.json', 'w') as f:
            json.dump(self.model_dirs, f, indent=4)

        with open(f'{self.root_folder}/explaination.json', 'w') as f:
            json.dump(self.explaination_dirs, f, indent=4)
        
        self.DF.to_csv(f'{self.root_folder}/result.csv')
    
    def trade(self):
        DF = self.DF.copy()
        DF = DF[['price_org', 'final_prob', 'dir_obs_price']]
        DF.dropna(subset=['price_org', 'final_prob', 'dir_obs_price'], inplace=True)
        y_prob_1 = DF['final_prob'].values
        y_prob_0 = 1 - y_prob_1
        y_prob = np.column_stack((y_prob_0, y_prob_1))
        y_true = DF['dir_obs_price'].values

        final = determine_reliable_predictions(y_prob, y_true)
        final = ['up' if x == 1 else 'down' for x in final]
        trades_df, metrics = simulate_trading(DF['price_org'].values, final, initial_balance=1000.0, trade_percent=0.1, sl_pct=0.015, tp_pct=0.02)
        return trades_df, metrics

    def evaluate(self):
        os.makedirs('saved/evaluation', exist_ok=True)
        if self.decomposition:
            r_preds = [f'pred_imf_{i}' for i in range(self.num_imfs)]
            r_trues = [f'imf_{i}' for i in range(self.num_imfs)]

            c_pred = 'pred_imf_0_class'
            c_true = 'dir_obs_imf_0'
        else:
            r_preds = ['pred_regre']
            r_trues = ['price_scaled']

            c_pred = 'pred_class'
            c_true = 'dir_obs_price'

        e_pred = 'final_pred'
        e_true = 'dir_obs_price'

        self.evaluation = {}
        for r_pred, r_true in zip(r_preds, r_trues):
            self.evaluation[f'{self.parts[1]} {r_true} {r_pred}'] = evaluate_regression_models(self.DF, r_true, r_pred, self.parts[1], self.dataset)
        self.evaluation[f'{self.parts[2]} {c_true} {c_pred}'] = evaluate_classifier_models(self.DF, c_true, c_pred, self.parts[2], self.dataset)
        self.evaluation[self.name] = evaluate_ensemble_models(self.DF, e_true, e_pred, self.name, self.dataset)
    
        return self.evaluation