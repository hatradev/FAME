import pandas as pd
import shap
import os

def write_model(x_train, y_train, driver, dir):
    os.makedirs(dir, exist_ok=True)
    model, path = driver(x_train, y_train, dir)
    return model, path

def build_explainer(model, x_train, x_test, save_dir, reshape_input):
    x_train_flat = x_train.reshape((x_train.shape[0], -1))

    def model_predict(x, model=model, reshape_input=reshape_input):
        if reshape_input:
            x = x.reshape((x.shape[0], x.shape[1], 1))
        return model.predict(x)
    
    mask = shap.maskers.Independent(x_train_flat[:50])
    explainer = shap.Explainer(model_predict, masker=mask, feature_names=[f'lag_{i+1}' for i in range(x_train.shape[1])])
    shap_values = explainer(x_test.reshape((x_test.shape[0], -1)))
    shap_df = pd.DataFrame(shap_values.values, columns=shap_values.feature_names)
    shap_df.to_csv(save_dir, index=False)
    return explainer