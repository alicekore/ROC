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
    CAE.CAE(x_train, x_test, path, params['epochs'], params['learning_rate'], params['units'])


if __name__ == '__main__':
    run_experiment('D:\Alisa\ROC\ROC repo\CAE\Experiments\Config1.json')
