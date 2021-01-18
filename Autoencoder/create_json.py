import json

params = {
    'ds_path': 'D:\Alisa\ROC\ROC repo\Autoencoder\Data Set\Random.json',
    'units': [10000, 5000, 1000],
    'learning_rate': 0.0005,
    'epochs': 20
}

with open('Experiments/Config1.json', 'w') as json_file:
    json.dump(params, json_file)
