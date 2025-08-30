import os
from utils.preprocess import read_x
from model.classifier import classifier_drivers
from model.regressor import regressor_drivers
from model.ensembler import ensembler_drivers
from xai.plot import create_explain_dict
from PyEMD import EMD

datasets = {}
data_folder = "data"

def load_datasets():
    for file in os.listdir(data_folder):
        if file.endswith('.csv'):
            name = file.split('/')[-1].split('.')[0].replace(' ', '_')
            datasets[name] = f'{data_folder}/{file}'

def run_pipeline(ModelClass,
                 data_dir, 
                 regressor_keys,
                 classifier_key,
                 ensembler_key,
                 decomposition=True):
    model = None
    DF = None
    led = {}
    csv_files = None

    dataset_name = data_dir.split('/')[-1].split('.')[0].replace(' ', '_')
    price_org = read_x(data_dir=data_dir)

    model_name = "EMD_" if decomposition else "_"
    regressors = []
    for regressor_key in regressor_keys:
        regressors.append(regressor_drivers[regressor_key])
        model_name += f'{regressor_key}_'
    
    classifier = classifier_drivers[classifier_key]
    model_name += f'{classifier_key}_'

    ensembler = ensembler_drivers[ensembler_key]
    model_name += f'{ensembler_key}'

    path = f'saved/{model_name}/{dataset_name}'
    if os.path.exists(path):
        df_path = explaination_path = model_path = None
        files = os.listdir(path)

        if 'result.csv' in files:
            df_path = os.path.join(path, 'result.csv')

        if 'explaination.json' in files:
            explaination_path = os.path.join(path, 'explaination.json')

        if 'model.json' in files:
            model_path = os.path.join(path, 'model.json')

        model = ModelClass(model_name, dataset_name, EMD() if decomposition else None)
        model.load(df_path, explaination_path, model_path)
    else:
        model = ModelClass(model_name, dataset_name, EMD() if decomposition else None)
        model.fit(close_price=price_org,
                    regressor_drivers=regressors,
                    classifier_driver=classifier,
                    ensembler_driver=ensembler)
        
    DF = model.DF
    if model.explaination_dirs is not None:
        csv_files = model.explaination_dirs

    if csv_files is not None:
        led = create_explain_dict(csv_files)
    evaluation = model.evaluate()
    _, trade_result = model.trade()
    
    return model, DF, led, evaluation, trade_result