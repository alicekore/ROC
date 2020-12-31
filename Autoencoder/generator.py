from FyeldGenerator import generate_field
import matplotlib.pyplot as plt
import numpy as np

# Helper that generates power-law power spectrum
def Pkgen(n):
    def Pk(k):
        return np.power(k, -n)
    return Pk


# Draw samples from a normal distribution
def distrib(shape):
    a = np.random.normal(loc=0, scale=1, size=shape)
    b = np.random.normal(loc=0, scale=1, size=shape)
    return a + 1j * b


def createPic(size, spec, shape=(100, 100)):
    for i in range(size):
        field = generate_field(distrib, Pkgen(spec), shape)
        path = 'Data Set Tiff/Spectrum' + str(spec) + '/file' + str(i) + '.tiff'  # Set the path!
        plt.axis('off')
        plt.imshow(field, cmap='jet')
        plt.savefig(path, bbox_inches='tight', transparent=True, pad_inches=0)
        print('Done: ' + str(i))

def createArray(size, spec, shape=(100, 100)):

    for i in range(size):
        field = generate_field(distrib, Pkgen(spec), shape)
        print(field)


if __name__ == "__main__":
    # createPic(500, 1)
    # createPic(500, 2)
    # createPic(500, 3)
    # createPic(500, 4)
    # createPic(500, 5)
    # createPic(500, 6)
    # createPic(500, 7)
    # createPic(500, 8)
    # createPic(500, 9)
    # createPic(500, 10)
    createArray(1, 4, shape=(28, 28))

