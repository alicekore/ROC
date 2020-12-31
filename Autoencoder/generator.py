from FyeldGenerator import generate_field
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.inf)

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

# Create field and save as tiff file
def createPic(size, spec, shape=(100, 100)):
    for i in range(size):
        field = generate_field(distrib, Pkgen(spec), shape)
        path = 'Data Set Tiff/Spectrum' + str(spec) + '/file' + str(i) + '.tiff'  # Set the correct path!
        plt.axis('off')
        plt.imshow(field, cmap='jet')
        plt.savefig(path, bbox_inches='tight', transparent=True, pad_inches=0)
        print('Done: ' + str(i))

# Create field and save as txt file (one line - one field)
def createText(size, spec, shape=(100, 100)):
    file_name = 'Data Set/Spectrum' + str(spec) + '.txt' # Set the correct path!
    file = open(file_name, "a+")
    for i in range(size):
        field = generate_field(distrib, Pkgen(spec), shape)
        field_text =str(field).replace('\n', '')
        file.write(field_text + '\n')
        print('Done:' + str(i))
    file.close()





if __name__ == "__main__":
    createText(5000, 1)
    createText(5000, 2)
    # createText(5000, 3)
    # createText(5000, 4)
    createText(5000, 5)
    createText(5000, 6)
    createText(5000, 7)
    createText(5000, 8)
    createText(5000, 9)
    createText(5000, 10)

