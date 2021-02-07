import keras
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


def CAE(x_train, x_test, folder, epochs):
    input_img = keras.Input(shape=(100, 100, 1))

    x = layers.Conv2D(16, (3, 3), activation=LeakyReLU(alpha=0.3), padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation=LeakyReLU(alpha=0.3), padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation=LeakyReLU(alpha=0.3), padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

    x = layers.Conv2D(8, (3, 3), activation=LeakyReLU(alpha=0.3), padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(8, (3, 3), activation=LeakyReLU(alpha=0.3), padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(16, (3, 3), activation=LeakyReLU(alpha=0.3))(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(1, (3, 3), activation=LeakyReLU(alpha=0.3), padding='same')(x)

    autoencoder = keras.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')

    autoencoder.fit(x_train, x_train,
                    epochs=epochs,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(x_test, x_test))

    decoded_imgs = autoencoder.predict(x_train)

    num_images_to_show = 5
    plot_ind = 1
    for im_ind in range(num_images_to_show):
        rand_ind = numpy.random.randint(low=0, high=x_train.shape[0])

        plt.subplot(num_images_to_show, 3, plot_ind)
        plt.axis('off')
        plt.imshow(x_train[rand_ind, :, :], cmap='jet')
        plot_ind = plot_ind + 1

        plt.subplot(num_images_to_show, 3, plot_ind)
        plt.axis('off')
        plt.imshow(decoded_imgs[rand_ind, :, :], cmap='jet')
        plot_ind = plot_ind + 1

        plt.subplot(num_images_to_show, 3, plot_ind)
        plt.axis('off')
        plt.imshow(numpy.subtract(x_train[rand_ind, :, :], decoded_imgs[rand_ind, :, :]),
                   cmap='jet')
        plot_ind = plot_ind + 1

    plt.savefig(str(folder) + '/random5.png')
    plt.clf()