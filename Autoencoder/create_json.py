import json

params = {
    'ds_path': 'D:\Alisa\ROC\ROC repo\Data Set\RandomMix.json',
    'units': [10000, 1],
    'learning_rate': 0.0005,
    'epochs': 100
}

with open('Experiments/Config8.json', 'w') as json_file:
    json.dump(params, json_file)
