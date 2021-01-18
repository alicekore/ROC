import autoencoder
import json

f = open('Experiments/Config1.json')
params = json.load(f)

x_train_orig, x_test_orig, x_train, x_test = autoencoder.prepareDataSet(params['ds_path'])
decoder, encoder, ae = autoencoder.autoencoder()
autoencoder.train_AE(ae, x_train, x_test, params['learning_rate'], params['epochs'])
autoencoder.evaluate_model(decoder, encoder, x_train_orig, x_train)

f.close()
