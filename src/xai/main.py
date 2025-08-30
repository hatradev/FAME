import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shap
from lime import lime_tabular
import joblib
import os

def read_model_obj_and_return_exp_pred_proba_use_shap(x_train, x_test, dumped_model_obj, loading_function=joblib.load, feature_names=None, y_test=None, file_shap_csv=None):
    model_obj=loading_function(dumped_model_obj)
    if feature_names is None:
        feature_names = [str(i) for i in range(x_train.shape[1])]
    explainer =   shap.TreeExplainer(model_obj, feature_names=feature_names)
    shap_values = explainer(x_test)
    shap.initjs()

    if file_shap_csv is not None:
        shap_df = pd.DataFrame(shap_values.values, columns=shap_values.feature_names)
        shap_df.to_csv(file_shap_csv, index=False)
    return model_obj.predict_proba(x_test)


def read_model_obj_and_return_exp_pred_proba(x_train, x_test, dumped_model_obj, loading_function=joblib.load, feature_names=None, y_test=None, save_dir=None):
    if save_dir != None:
        os.makedirs(save_dir, exist_ok=True)
    model_obj=loading_function(dumped_model_obj)

    if feature_names is None:
        feature_names = [str(i) for i in range(x_train.shape[1])]
    explainer = lime_tabular.LimeTabularExplainer(x_train, mode="classification", feature_names=feature_names, class_names=["Down","Up"])  
    explanation_predict_proba = []
    explanation_dicts = []
    for i in range(x_test.shape[0]):
        explanation = explainer.explain_instance(x_test[i],model_obj.predict_proba, num_features=x_train.shape[1])
        explanation_predict_proba.append(explanation.predict_proba)
        exp_dict = dict(explanation.as_list())
        explanation_dicts.append(exp_dict)

        fig = explanation.as_pyplot_figure()
        fig.savefig(f"{save_dir}lime_explanation_{i}.png", bbox_inches='tight')
        plt.close(fig)
    return np.array(explanation_predict_proba), explanation_dicts

def determine_reliable_predictions(exp_pred_proba_arr, y_test, reliability_level=0.5):
    rel_pred=[]
    rl=reliability_level
    for i, j  in zip(exp_pred_proba_arr, y_test):
        if i[0] >= rl and j == 0:
            rel_pred.append(1)
        elif i[1] >= rl and j == 1:
            rel_pred.append(1)
        elif i[0] >= rl and j == 1:
            rel_pred.append(0)
        elif i[1] >= rl and j == 0:
            rel_pred.append(0)
        else:
            rel_pred.append(np.NaN)
    return rel_pred

def compute_hitrate(rel_pred_list):
    l=rel_pred_list
    nort= l.count(1)
    norf= l.count(0)
    norp = nort + norf

    if norp == 0:
        return 0,0
    hitrate = nort / norp
    return norp,hitrate

def compute_reliability_vs_norp_tradeoff(ll, ul, step, exp_pred_proba_arr, y_test):
    output_df=pd.DataFrame()
    norps=[]
    hitrates=[]
    
    try:
        np_range=np.arange(ll, ul, step)
        list_range=list(np_range)
        list_range.append(ul)

        for i in list_range:
            rel_pred=determine_reliable_predictions(exp_pred_proba_arr, y_test, i)
            t=compute_hitrate(rel_pred)
            norps.append(t[0])
            hitrates.append(t[1])

        output_df['Reliability Levels']=list_range
        output_df['Number of Rel Preds']=norps
        output_df['Hitrates']=hitrates
    except:
        if((np.isreal(ll) and np.isreal(ul) and np.isreal(step)) is not True):
            print("Lower limit, upper limit and step values must be real numbers!")
            raise ValueError
        elif( ll >= ul):
            print("Lower limit cannot be greater than upper limit!")
            raise ValueError
        elif(ll < 0.5):
            print("Lower limit cannot be smaller than 0.5!")
            raise ValueError
        elif(ul > 1.0):
            print("Upper limit cannot be greater than 1.0!")
            raise ValueError
        elif((0.1 >= step >= 0.01) is not True):
            print("Step must be in-between 0.01 and 0.1 range!")
            raise ValueError
        else:
            raise ValueError
    else:
        pass

    return output_df