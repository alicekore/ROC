import tensorflow.keras.layers
import tensorflow.keras.models
import tensorflow.keras.optimizers
import tensorflow.keras.datasets
import numpy
import matplotlib.pyplot
import os
import codecs, json
from sklearn.model_selection import train_test_split


# Loads the dataset from json file and returns it as reshaped numpy array
def getDataSet():
    path = input("Enter the path to data set json file: ")
    assert os.path.exists(path), "File not found at " + str(path)

    obj_text = codecs.open(path, 'r', encoding='utf-8').read()
    raw = json.loads(obj_text)
    set = numpy.array(raw)
    return set


def autoencoder():
    x = tensorflow.keras.layers.Input(shape=(10000), name="encoder_input")

    encoder_dense_layer1 = tensorflow.keras.layers.Dense(units=5000, name="encoder_dense_1")(x)
    encoder_activ_layer1 = tensorflow.keras.layers.LeakyReLU(name="encoder_leakyrelu_1")(encoder_dense_layer1)

    encoder_dense_layer2 = tensorflow.keras.layers.Dense(units=1000, name="encoder_dense_2")(encoder_activ_layer1)
    encoder_output = tensorflow.keras.layers.LeakyReLU(name="encoder_output")(encoder_dense_layer2)

    encoder = tensorflow.keras.models.Model(x, encoder_output, name="encoder_model")
    # encoder.summary()


    decoder_input = tensorflow.keras.layers.Input(shape=(1000), name="decoder_input")

    decoder_dense_layer1 = tensorflow.keras.layers.Dense(units=5000, name="decoder_dense_1")(decoder_input)
    decoder_activ_layer1 = tensorflow.keras.layers.LeakyReLU(name="decoder_leakyrelu_1")(decoder_dense_layer1)

    decoder_dense_layer2 = tensorflow.keras.layers.Dense(units=10000, name="decoder_dense_2")(decoder_activ_layer1)
    decoder_output = tensorflow.keras.layers.LeakyReLU(name="decoder_output")(decoder_dense_layer2)

    decoder = tensorflow.keras.models.Model(decoder_input, decoder_output, name="decoder_model")
    # decoder.summary()


    ae_input = tensorflow.keras.layers.Input(shape=(10000), name="AE_input")
    ae_encoder_output = encoder(ae_input)
    ae_decoder_output = decoder(ae_encoder_output)

    ae = tensorflow.keras.models.Model(ae_input, ae_decoder_output, name="AE")
    # ae.summary()

    # AE Compilation
    ae.compile(loss="mse", optimizer=tensorflow.keras.optimizers.Adam(lr=0.0005))

    # Preparing the dataset
    orig = getDataSet()
    x_train_orig, x_test_orig = train_test_split(orig, test_size=0.2)

    x_train = numpy.reshape(x_train_orig, newshape=(x_train_orig.shape[0], numpy.prod(x_train_orig.shape[1:])))
    x_test = numpy.reshape(x_test_orig, newshape=(x_test_orig.shape[0], numpy.prod(x_test_orig.shape[1:])))

    # Training AE
    ae.fit(x_train, x_train, epochs=20, batch_size=256, shuffle=True,
           validation_data=(x_test, x_test))  # validation data <> validation! (validation set)

    encoded_images = encoder.predict(x_train)
    decoded_images = decoder.predict(encoded_images)
    decoded_images_orig = numpy.reshape(decoded_images, newshape=(decoded_images.shape[0], 100, 100))

    num_images_to_show = 5
    for im_ind in range(num_images_to_show):
        plot_ind = im_ind * 2 + 1
        rand_ind = numpy.random.randint(low=0, high=x_train.shape[0])
        matplotlib.pyplot.subplot(num_images_to_show, 2, plot_ind)
        matplotlib.pyplot.axis('off')
        matplotlib.pyplot.imshow(x_train_orig[rand_ind, :, :], cmap='jet')
        matplotlib.pyplot.subplot(num_images_to_show, 2, plot_ind + 1)
        matplotlib.pyplot.axis('off')
        matplotlib.pyplot.imshow(decoded_images_orig[rand_ind, :, :], cmap='jet')
    matplotlib.pyplot.show()


if __name__ == "__main__":
    autoencoder()