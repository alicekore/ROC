import json

params = {
    'ds_path': 'D:\Alisa\ROC\ROC repo\Data Set\RandomMix.json',
    'units': [32, 16, 8],
    'epochs': 10,
    'learning_rate': 0.0005,
}

with open('D:\Alisa\ROC\ROC repo\CAE\Experiments\Config1.json', 'w') as json_file:
    json.dump(params, json_file)