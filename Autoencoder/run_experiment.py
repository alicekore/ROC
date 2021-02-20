import autoencoder
import json
import os
from datetime import datetime
import matplotlib.pyplot
import shutil
import numpy as np


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
    history = autoencoder.train_AE(ae, x_train, x_test, params['learning_rate'], params['epochs'], path)
    avg_error = autoencoder.evaluate_model(decoder, encoder, x_train_orig, x_train, path)
    loss = history.history['loss']
    val_loss = history.history['val_loss']

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
        "val_loss": val_loss,
        "avg_error": avg_error
    })
    with open('D:/Alisa/ROC/ROC repo/Autoencoder/Experiments/results.json', 'w') as outfile:
        json.dump(data, outfile)

    f.close()

    dim = np.arange(1, len(history.history['loss']) + 1)
    matplotlib.pyplot.plot(dim, history.history['loss'])
    matplotlib.pyplot.xlim(5)
    matplotlib.pyplot.plot(dim, history.history['val_loss'])
    matplotlib.pyplot.ylabel('loss')
    matplotlib.pyplot.xlabel('epoch')
    matplotlib.pyplot.legend(['train', 'test'], loc='upper left')
    matplotlib.pyplot.savefig(path + '/loss.png', dpi=300)
    matplotlib.pyplot.clf()

if __name__ == '__main__':
    run_experiment('D:\Alisa\ROC\ROC repo\Autoencoder\Experiments\Config9.json')
