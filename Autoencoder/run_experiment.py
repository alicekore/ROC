import autoencoder
import json
import os
from datetime import datetime
import shutil


def run_experiment(config):
    # Create new experiment folder
    path = "D:/Alisa/ROC/ROC repo/Autoencoder/Experiments/" + datetime.now().strftime('%Y.%m.%d %H-%M-%S')
    os.mkdir(path)
    # Copy config file to experiment folder
    shutil.copy2(config, path)
    # Open json
    f = open(str(config))
    params = json.load(f)

    x_train_orig, x_test_orig, x_train, x_test = autoencoder.prepareDataSet(params['ds_path'])
    decoder, encoder, ae = autoencoder.autoencoder_generic(params['units'])
    autoencoder.train_AE(ae, x_train, x_test, params['learning_rate'], params['epochs'])
    autoencoder.evaluate_model(decoder, encoder, x_train_orig, x_train, path)

    f.close()


if __name__ == '__main__':
    run_experiment('D:\Alisa\ROC\ROC repo\Autoencoder\Experiments\Config1.json')