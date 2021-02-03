import numpy as np
import matplotlib.pyplot as plt
import json

np.random.seed(111)
data = json.load(open('D:/Alisa/ROC/ROC repo/Autoencoder/Experiments/results.json'))
results = []
labels = []
for i in range(len(data)):
    results.append(data[i]["val_loss"])
    labels.append(str(data[i]["units"]))

plt.figure(figsize=(15, 15))
plt.xlabel('Units in layers')
plt.ylabel('Val_acc')
plt.boxplot(x=results, labels=labels)
plt.show()
