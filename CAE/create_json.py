import json

params = {
    'ds_path': 'D:\Alisa\ROC\ROC repo\Data Set\RandomMix.json',
    'epochs': 100,
}

with open('D:\Alisa\ROC\ROC repo\CAE\Experiments\Config1.json', 'w') as json_file:
    json.dump(params, json_file)