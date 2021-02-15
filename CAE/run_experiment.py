import CAE
import json
import os
from datetime import datetime
import shutil
import numpy as np


def run_experiment(config):
    time = datetime.now().strftime('%Y.%m.%d %H-%M-%S')
    path = "D:/Alisa/ROC/ROC repo/CAE/Experiments/" + str(time)
    os.mkdir(path)
    # Copy config file to experiment folder
    shutil.copy2(config, path)
    # Open json
    f = open(str(config))
    params = json.load(f)

    x_train_orig, x_test_orig, x_train, x_test = CAE.prepareDataSet(params['ds_path'])
    history = CAE.CAE(x_train, x_test, path, params['epochs'], params['learning_rate'], params['units'])
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    data = json.load(open('D:/Alisa/ROC/ROC repo/CAE/Experiments/results.json'))
    if type(data) is dict:
        data = [data]
    data.append({
        "date": time,
        "ds_path": params['ds_path'],
        "units": params['units'],
        "learning rate": params['learning_rate'],
        "epochs": params['epochs'],
        "loss": loss,
        "val_loss": val_loss
    })
    with open('D:/Alisa/ROC/ROC repo/CAE/Experiments/results.json', 'w') as outfile:
        json.dump(data, outfile)

    f.close()

if __name__ == '__main__':
    run_experiment('D:\Alisa\ROC\ROC repo\CAE\Experiments\Config1.json')
