from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import joblib
import os
from model.config import ensembler_config as config

def xgboost_classifier_driver(x_train, y_train, dump_file):
    model = XGBClassifier(n_estimators=config["XGB"]["NUM_ESTIMATORS"], max_depth=config["XGB"]["MAX_DEPTH"], learning_rate=config["XGB"]["LEARNING_RATE"], use_label_encoder=config["XGB"]["USE_LABEL_ENCODER"], eval_metric=config["XGB"]["EVAL_METRIC"])
    model.fit(x_train, y_train)
    path = os.path.join(dump_file, "model")
    joblib.dump(model, path)
    return model, path

def rf_classifier_driver(x_train, y_train, dump_file):
    model = RandomForestClassifier(n_estimators=config["RF"]["NUM_ESTIMATORS"], max_depth=config["RF"]["MAX_DEPTH"], random_state=config["RF"]["RANDOM_STATE"])
    model.fit(x_train, y_train)
    path = os.path.join(dump_file, "model")
    joblib.dump(model, path)
    return model, path

def lightgbm_classifier_driver(x_train, y_train, dump_file):
    model = LGBMClassifier(n_estimators=config["LGBM"]["NUM_ESTIMATORS"], max_depth=config["LGBM"]["MAX_DEPTH"], learning_rate=config["LGBM"]["LEARNING_RATE"])
    model.fit(x_train, y_train)
    path = os.path.join(dump_file, "model")
    joblib.dump(model, path)
    return model, path

def catboost_classifier_driver(x_train, y_train, dump_file):
    model = CatBoostClassifier(iterations=config["CB"]["ITERATIONS"], learning_rate=config["CB"]["LEARNING_RATE"], depth=config["CB"]["DEPTH"])
    model.fit(x_train, y_train)
    path = os.path.join(dump_file, "model")
    joblib.dump(model, path)
    return model, path

ensembler_drivers = {
    "XGB": xgboost_classifier_driver,
    "RF": rf_classifier_driver,
    "LGBM": lightgbm_classifier_driver,
    "CB": catboost_classifier_driver,
}