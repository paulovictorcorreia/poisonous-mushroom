import json
import joblib
import pandas as pd
from sklearn.metrics import f1_score
import os
import yaml
import argparse
from typing import Text

def evaluate(config: Text) -> None:
    with open(config) as conf_file:
        config = yaml.safe_load(conf_file)

    model_path = config['train']['model_path']
    model = joblib.load(model_path)
    train_file = config['preprocessing']['trainset_path']
    val_file = config['preprocessing']['valset_path']
    id_column = config['metadata']['id_col']
    target_column = config['metadata']['target_col']



    train_data = pd.read_csv(train_file, index_col=id_column)
    x_train = train_data.drop(target_column, axis=1)
    y_train = train_data[target_column].values
    y_train_pred = model.predict(x_train)
    f1_train = f1_score(y_train, y_train_pred)

    

    val_data = pd.read_csv(val_file, index_col=id_column)
    x_val = val_data.drop(target_column, axis=1)
    y_val = val_data[target_column].values
    y_pred = model.predict(x_val)
    f1_val = f1_score(y_val, y_pred)

    report = {
        'train':{
            'f1': f1_train,
            # 'actual': list(y_train),
            # 'predicted': list(y_train_pred)
        },
        'val':{
            'f1': f1_val,
            # 'actual': list(y_val),
            # 'predicted': list(y_pred)
        }
    }
    report_path = os.path.join(config['evaluate']['reports_dir'], config['evaluate']['metrics_file'])
    with open(report_path, 'w+') as file:
        json.dump(report, file)

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    evaluate(config=args.config)