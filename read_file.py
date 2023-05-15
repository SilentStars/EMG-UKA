import os
import numpy as np
import matplotlib.pyplot as plt

def normalize(x):
    x = np.asarray(x)
    arr = (x - x.min()) / (int(x.max()) - int(x.min()))
    return np.subtract(np.multiply(arr, 2), 1)


file_path = 'EMG-UKA-Full-Corpus/emg/501/001/e07_501_001_0020.adc'
f = open(file_path, 'rb')
f = np.fromfile(file_path, dtype='<i2').astype('int32')
f = np.reshape(f, (int(len(f) / 7), 7)).T
res = []
for i in range(0, len(f)):
    res.append(normalize(f[i]))
    
res = np.array(res)
plt.plot(range(0, len(res[0])), res[0])
plt.show()

print(f)