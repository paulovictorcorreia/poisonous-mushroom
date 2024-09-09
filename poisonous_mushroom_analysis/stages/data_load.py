import os
import gdown
import zipfile
import yaml
import argparse

def extract_datasets(config_path: str) -> None:
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    raw_data_path = config['data_load']['raw_path']
    filename = config['data_load']['compact_name']
    file_id = config['data_load']['file_id']
    download_url = f'https://drive.google.com/uc?id={file_id}'
    output = os.path.join(raw_data_path, filename)
    print(f'Downloading to {output}')
    gdown.download(download_url, output, quiet=False)

    print('Extracting datasets')
    zip_file_path = os.path.join(raw_data_path, filename)
    processed_path = config['data_decompress']['processed_path']
    os.makedirs(processed_path, exist_ok=True)
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(processed_path)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    extract_datasets(config_path=args.config)