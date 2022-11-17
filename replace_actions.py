import os
import glob
import pickle
from src.utils.utils import load_yaml
from src.data.predict import replace_actions


dataset_pattern = os.path.join('artifacts', 'dataset', 'predict_*')
prediction_config = load_yaml(os.path.join('config', 'prediction_config.yaml'))
dataset_file = glob.glob(dataset_pattern)

for dataset in dataset_file:
    with open(dataset, 'rb') as f:
        batch = pickle.load(f)

    output_path = dataset.replace(prediction_config['data_name'], prediction_config['data_name'] + '_new_action')
    replace_actions(prediction_config, batch)

    with open(output_path, 'wb') as f:
        pickle.dump(batch, f)

