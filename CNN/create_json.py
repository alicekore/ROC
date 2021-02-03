import json

params = {
    'ds_path': 'D:\Alisa\ROC\ROC repo\Data Set\RandomMix.json',
}

with open('D:/Alisa/ROC/ROC repo/CNN/Experiments/Config1.json', 'w') as json_file:
    json.dump(params, json_file)