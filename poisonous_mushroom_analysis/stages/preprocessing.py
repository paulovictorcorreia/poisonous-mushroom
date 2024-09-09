import yaml
from typing import Text
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
import argparse


def split_and_preprocess_data(config: Text) -> None:

    with open(config) as conf_file:
        config = yaml.safe_load(conf_file)

    train_file = config['preprocessing']['original_train']
    test_file = config['preprocessing']['original_test']
    id_column = config['metadata']['id_col']
    target_column = config['metadata']['target_col']

    train_data = pd.read_csv(train_file, index_col=id_column)
    train_data.drop(columns=['stem-height', 'stem-width', 'cap-diameter'], inplace=True)
    

    test_data = pd.read_csv(test_file, index_col=id_column)
    test_data.drop(columns=['stem-height', 'stem-width', 'cap-diameter'], inplace=True)


    train_data[target_column] = train_data[target_column].replace({'e':0.0, 'p':1.0})

    x_train, x_val, y_train, y_val = train_test_split(
        train_data.drop(target_column, axis=1), train_data[target_column],
        test_size=config['preprocessing']['test_size'],
        random_state=config['base']['random_state']
    )

    columns = x_train.columns.to_list()
    idx_train = x_train.index
    idx_val = x_val.index
    idx_test = test_data.index


    imputer = SimpleImputer(strategy='most_frequent')
    x_train = imputer.fit_transform(x_train)
    x_val = imputer.transform(x_val)
    x_test = imputer.transform(test_data)

    ordinal_encoding = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    x_train = ordinal_encoding.fit_transform(x_train)
    x_val = ordinal_encoding.transform(x_val)
    x_test = ordinal_encoding.transform(x_test)

    x_train = pd.DataFrame(x_train, columns=columns, index=idx_train)
    x_val = pd.DataFrame(x_val, columns=columns, index=idx_val)
    x_test = pd.DataFrame(x_test, columns=columns, index=idx_test)

    train_data = pd.concat([x_train, y_train], axis=1)
    val_data = pd.concat([x_val, y_val], axis=1)

    train_data.to_csv(config['preprocessing']['trainset_path'])
    val_data.to_csv(config['preprocessing']['valset_path'])
    x_test.to_csv(config['preprocessing']['testset_path'])

    pass


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    split_and_preprocess_data(config=args.config)