import numpy as np
import matplotlib.pyplot as plt
import random
import codecs
import json
np.set_printoptions(threshold=np.inf)


def gaussian_2d(center=(50, 50), sig=10, image_size=(100, 100)):
    """
    It produces single gaussian at expected center
    :param center:  the mean position (X, Y) - where high value expected
    :param sig: The sigma value
    :param image_size: The total image size (width, height)
    """
    x_axis = np.linspace(0, image_size[0] - 1, image_size[0]) - center[0]
    y_axis = np.linspace(0, image_size[1] - 1, image_size[1]) - center[1]
    xx, yy = np.meshgrid(x_axis, y_axis)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
    return kernel


def overlap_gauss(image_size=(100, 100)):
    result = np.zeros(image_size)
    max = image_size[0]
    loop_size = random.randint(1, 10)
    for i in range(loop_size):
        center = (random.randrange(max), random.randrange(max))
        sig = random.randrange(5, max / 2)
        gauss = gaussian_2d(center, sig)
        result = np.add(result, gauss)
    return result


def create_data_set(size, shape=(100, 100)):
    file_name = 'Data Set/Random' + '.json'  # Set the correct path!
    origs = np.empty((size, shape[0], shape[1]))
    for i in range(size):
        field = overlap_gauss(shape)
        origs[i] = field
        print('Done:' + str(i))
    set = origs.tolist()  # nested lists with same data, indices
    json.dump(set, codecs.open(file_name, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)


def get_random_image():
    plt.imshow(overlap_gauss(), cmap='jet')
    plt.show()


if __name__ == '__main__':
    create_data_set(10000)
