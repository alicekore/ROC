import numpy as np
import matplotlib.pyplot as plt
import random
import codecs
import json
np.set_printoptions(threshold=np.inf)


def gaussian_2d(center, sig, image_size=(100, 100)):
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


# Creates gauss with SPECIFIED parameters
def generic_overlap_gauss(centers: tuple, sigs: tuple, image_size=(100, 100), normalize=False, negative=False):
    if len(centers) != len(sigs):
        raise ValueError("Amount of center values and sigma values are different!")
    result = np.zeros(image_size)
    for i in range(len(sigs)):
        gauss = gaussian_2d(centers[i], sigs[i])
        result = np.add(result, gauss)
    if normalize:
        norm = np.linalg.norm(result)
        result = result / norm
    if negative:
        result = np.negative(result)
    return result


# Creates RANDOM gauss
def overlap_gauss(image_size=(100, 100), normalize=False, negative=False):
    result = np.zeros(image_size)
    max = image_size[0]
    loop_size = random.randint(1, 10)
    for i in range(loop_size):
        center = (random.randrange(max), random.randrange(max))
        sig = random.randrange(int(max / 10), int(max / 2))
        gauss = gaussian_2d(center, sig)
        result = np.add(result, gauss)
    if normalize:
        norm = np.linalg.norm(result)
        result = result / norm
    if negative:
        result = np.negative(result)
    return result


def create_data_set(ds_size, file_name, image_size=(100, 100), normalize=False, negative=False):
    file_name = 'Data Set/' + str(file_name) + '.json'
    origs = np.empty((ds_size, image_size[0], image_size[1]))
    for i in range(ds_size):
        field = overlap_gauss(image_size, normalize, negative)
        origs[i] = field
        print('Done:' + str(i))
    ds = origs.tolist()  # nested lists with same data, indices
    json.dump(ds, codecs.open(file_name, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)


def get_random_image(normalize=False, negative=False):
    plt.imshow(overlap_gauss(normalize=normalize, negative=negative), cmap='jet')
    plt.show()


if __name__ == '__main__':
    plt.imshow(generic_overlap_gauss(centers=((50,50),(20,80)), sigs=(30, 10)), cmap='jet')
    plt.show()
