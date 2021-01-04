from FyeldGenerator import generate_field
import matplotlib.pyplot as plt
import numpy as np
import codecs, json
import random
np.set_printoptions(threshold=np.inf)
NOISE = [1, 2]
CLOUDS = [3, 4, 5, 6]
BUBBLES = [7, 8, 9, 10]

# Helper that generates power-law power spectrum
def Pkgen(n):
    def Pk(k):
        return np.power(k, -n)
    return Pk


# Draws samples from a normal distribution
def distrib(shape):
    a = np.random.normal(loc=0, scale=1, size=shape)
    b = np.random.normal(loc=0, scale=1, size=shape)
    return a + 1j * b

# Creates fields and saves them as tiff files
def createPic(size, spec, shape=(100, 100)):
    for i in range(size):
        field = generate_field(distrib, Pkgen(spec), shape)
        path = 'Data Set Tiff/Spectrum' + str(spec) + '/file' + str(i) + '.tiff'  # Set the correct path!
        plt.axis('off')
        plt.imshow(field, cmap='jet')
        plt.savefig(path, bbox_inches='tight', transparent=True, pad_inches=0)
        print('Done: ' + str(i))

# Creates fields and saves them in json file
def createText(size, spec, shape=(100, 100)):
    file_name = 'Data Set/Bubbles' + '.json' # Set the correct path!
    origs = np.empty((size, shape[0], shape[1]))
    for i in range(size):
        field = generate_field(distrib, Pkgen(random.choice(spec)), shape)
        origs[i] = field
        print('Done:' + str(i))
    set = origs.tolist() # nested lists with same data, indices
    json.dump(set, codecs.open(file_name, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)

if __name__ == "__main__":
    createText(10000, BUBBLES)


