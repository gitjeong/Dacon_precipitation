import numpy as np
import glob

train_files = glob.glob('./data_rescaled/train/*.npy')

data = np.empty([62735,12,12,5])
idx = 0
for file in train_files:
    arr = np.load(file)
    data[idx,:,:,:] = arr.reshape([1,12,12,5])
    idx = idx + 1

np.save('data_rescaled/data', data)

