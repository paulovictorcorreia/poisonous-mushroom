import joblib
from typing import Text
import yaml
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import argparse

def train(config: Text):

    with open(config) as conf_file:
        config = yaml.safe_load(conf_file)

    train_data = pd.read_csv(config['preprocessing']['trainset_path'],
                             index_col=config['metadata']['id_col'])

    x_train = train_data.drop("class", axis=1)
    y_train = train_data['class']



    rfc = RandomForestClassifier(
        n_estimators=config['train']['random_forest']['n_estimators'],
        max_depth=config['train']['random_forest']['max_depth'],
        random_state=config['base']['random_state'],
        n_jobs=config['base']['n_jobs']
    )

    rfc.fit(x_train, y_train)

    models_path = config['train']['model_path']

    joblib.dump(rfc, models_path)

    pass

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    train(args.config)