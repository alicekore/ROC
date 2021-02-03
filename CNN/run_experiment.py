import CNN
import json
import os
from datetime import datetime
import shutil


def run_experiment(config):
    time = datetime.now().strftime('%Y.%m.%d %H-%M-%S')
    path = "D:/Alisa/ROC/ROC repo/CNN/Experiments/" + str(time)
    os.mkdir(path)
    # Copy config file to experiment folder
    shutil.copy2(config, path)
    # Open json
    f = open(str(config))
    params = json.load(f)

    x_train_orig, x_test_orig, train_images, train_labels, test_images, test_labels = CNN.prepareDataSet(params['ds_path'])
    model = CNN.CNN()
    history = CNN.train_CNN(model, train_images, train_labels, test_images, test_labels)



if __name__ == '__main__':
    run_experiment('D:\Alisa\ROC\ROC repo\CNN\Experiments\Config1.json')
