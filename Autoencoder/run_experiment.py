import autoencoder
import json
import os
from datetime import datetime
import shutil


def run_experiment(config):
    # Create new experiment folder
    time = datetime.now().strftime('%Y.%m.%d %H-%M-%S')
    path = "D:/Alisa/ROC/ROC repo/Autoencoder/Experiments/" + str(time)
    os.mkdir(path)
    # Copy config file to experiment folder
    shutil.copy2(config, path)
    # Open json
    f = open(str(config))
    params = json.load(f)

    x_train_orig, x_test_orig, x_train, x_test = autoencoder.prepareDataSet(params['ds_path'])
    decoder, encoder, ae = autoencoder.autoencoder_generic(params['units'])
    loss, val_loss = autoencoder.train_AE(ae, x_train, x_test, params['learning_rate'], params['epochs'], path)
    autoencoder.evaluate_model(decoder, encoder, x_train_orig, x_train, path)

    data = json.load(open('D:/Alisa/ROC/ROC repo/Autoencoder/Experiments/results.json'))
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
    with open('D:/Alisa/ROC/ROC repo/Autoencoder/Experiments/results.json', 'w') as outfile:
        json.dump(data, outfile)

    f.close()


if __name__ == '__main__':
    run_experiment('D:\Alisa\ROC\ROC repo\Autoencoder\Experiments\Config9.json')
