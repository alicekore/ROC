import autoencoder
import json
import os
from datetime import datetime

# Create new experiment directory
path = "D:/Alisa/ROC/ROC repo/Autoencoder/Experiments/" + datetime.now().strftime('%Y.%m.%d %H-%M-%S')
os.mkdir(path)

f = open('Experiments/Config1.json')
params = json.load(f)

x_train_orig, x_test_orig, x_train, x_test = autoencoder.prepareDataSet(params['ds_path'])
decoder, encoder, ae = autoencoder.autoencoder_generic(params['units'])
autoencoder.train_AE(ae, x_train, x_test, params['learning_rate'], params['epochs'])
autoencoder.evaluate_model(decoder, encoder, x_train_orig, x_train)

f.close()
