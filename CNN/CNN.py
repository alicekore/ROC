import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy
import os
import codecs
from sklearn.model_selection import train_test_split
import json


# Tutorial: https://www.tensorflow.org/tutorials/images/cnn

# Loads the dataset from json file
def prepareDataSet(path):
    path = str(path)
    assert os.path.exists(path), "File not found at " + str(path)

    obj_text = codecs.open(path, 'r', encoding='utf-8').read()
    raw = json.loads(obj_text)
    set = numpy.array(raw)

    x_train_orig, x_test_orig = train_test_split(set, test_size=0.2)
    # Add a channels dimension
    x_train = x_train_orig[..., tf.newaxis].astype("float32")
    x_test = x_test_orig[..., tf.newaxis].astype("float32")
    # Dummies as labels ?
    y_train = numpy.ones((len(x_train), 1))
    y_test = numpy.ones((len(x_test), 1))
    return x_train_orig, x_test_orig, x_train, y_train, x_test, y_test


def CNN():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    return model


def train_CNN(model, train_images, train_labels, test_images, test_labels):
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=10,
                        validation_data=(test_images, test_labels))

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    return history
