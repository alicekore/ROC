import keras
import tensorflow.keras.datasets
import tensorflow.keras.layers
import tensorflow.keras.models
import tensorflow.keras.optimizers
from keras import layers
import numpy
import os
import codecs
from sklearn.model_selection import train_test_split
import json
import matplotlib.pyplot as plt
from keras.layers.advanced_activations import LeakyReLU


def prepareDataSet(path):
    path = str(path)
    assert os.path.exists(path), "File not found at " + str(path)

    obj_text = codecs.open(path, 'r', encoding='utf-8').read()
    raw = json.loads(obj_text)
    set = numpy.array(raw)

    x_train_orig, x_test_orig = train_test_split(set, test_size=0.2)

    x_train = numpy.reshape(x_train_orig, (len(x_train_orig), 100, 100, 1))
    x_test = numpy.reshape(x_test_orig, (len(x_test_orig), 100, 100, 1))

    return x_train_orig, x_test_orig, x_train, x_test


def CAE(x_train, x_test, folder, epochs, learning_rate, units):
    input_img = keras.Input(shape=(100, 100, 1))

    x = layers.Conv2D(units[0], (3, 3), padding='same')(input_img)
    x = LeakyReLU(alpha=0.3)(x)
    for i in range(1, len(units)):
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(units[i], (3, 3), padding='same')(x)
        x = LeakyReLU(alpha=0.3)(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

    runits = units[::-1]  # Reverse array
    x = layers.Conv2D(runits[0], (3, 3), padding='same')(encoded)
    x = LeakyReLU(alpha=0.3)(x)
    x = layers.UpSampling2D((2, 2))(x)
    for i in range(1, len(runits)):
        x = layers.Conv2D(runits[i], (3, 3), padding='same')(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = layers.UpSampling2D((2, 2))(x)

    decoded = layers.Conv2D(1, (3, 3), padding='same')(x)
    decoded = LeakyReLU(alpha=0.3)(decoded)
    crop_factor = int((numpy.ceil(100 / 2 ** len(units)) * 2 ** len(units) - 100) / 2)
    decoded_cropping = layers.Cropping2D((crop_factor, crop_factor))(decoded)

    autoencoder = keras.Model(input_img, decoded_cropping)
    autoencoder.compile(optimizer=keras.optimizers.Adam(lr=learning_rate), loss='mse')
    my_callbacks = [
        tensorflow.keras.callbacks.EarlyStopping(patience=10),
        tensorflow.keras.callbacks.ModelCheckpoint(filepath=str(folder) + '/model.{epoch:02d}-{val_loss:.10f}.h5',
                                                   monitor='loss', verbose=1,
                                                   save_best_only=True, mode='auto', period=1),
        # tensorflow.keras.callbacks.TensorBoard(log_dir=str(folder), profile_batch=100000000),
    ]
    history = autoencoder.fit(x_train, x_train,
                              epochs=epochs,
                              batch_size=128,
                              shuffle=True,
                              validation_data=(x_test, x_test),
                              callbacks=my_callbacks)

    decoded_imgs = autoencoder.predict(x_train)

    num_images_to_show = 5
    plot_ind = 1
    for im_ind in range(num_images_to_show):
        rand_ind = numpy.random.randint(low=0, high=x_train.shape[0])

        plt.suptitle("Original          Recreated           Error", fontsize=12)

        plt.subplot(num_images_to_show, 3, plot_ind)
        plt.axis('off')
        plt.imshow(x_train[rand_ind, :, :], cmap='jet')
        plot_ind = plot_ind + 1

        plt.subplot(num_images_to_show, 3, plot_ind)
        plt.axis('off')
        plt.imshow(decoded_imgs[rand_ind, :, :], cmap='jet')
        plot_ind = plot_ind + 1

        diff = numpy.subtract(x_train[rand_ind, :, :], decoded_imgs[rand_ind, :, :])
        error = numpy.mean(numpy.absolute(diff))
        plt.subplot(num_images_to_show, 3, plot_ind)
        plt.axis('off')
        plt.title('{:.2f}e-5'.format(error/1e-5), fontsize=8)
        plt.imshow(diff, cmap='gray')
        plot_ind = plot_ind + 1

    plt.subplots_adjust(hspace=0.5, wspace=-0.5)
    plt.savefig(str(folder) + '/random5.png', dpi=300)
    plt.clf()

    error = 0
    for i in range(len(x_train)):
        diff = numpy.subtract(x_train[i, :, :], decoded_imgs[i, :, :])
        error = error + numpy.mean(numpy.absolute(diff))
    avg_error = error / len(x_train)
    return history, avg_error
