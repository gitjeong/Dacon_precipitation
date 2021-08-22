import numpy as np
import cv2
import glob
import tqdm

"""
test_files = sorted(glob.glob('./data/test/*.npy'))

arr_ = np.empty([12,12,4])
for file in tqdm.tqdm(test_files, desc='test'):
    arr = np.load(file)
    for idx in range(4):
        arr_[:,:,idx] = cv2.resize(arr[:,:,idx].reshape(120,120), dsize=(12, 12), interpolation=cv2.INTER_AREA)
    np.save('./data_rescaled/test/'+file.replace('./data/test/',''), arr_)
"""

files = np.load('./dacon_pcp_simpleDNN_result.npy')
num_files = files.shape[0]

Pred_result = []
for idx in range(num_files):
    arr_ = cv2.resize(files[idx,:].reshape(12,12), dsize=(120,120), interpolation=cv2.INTER_CUBIC)
    arr_ = arr_.reshape(1,120,120)
    Pred_result.append(arr_)
Pred_result = np.array(Pred_result)

np.save('./dacon_pcp_simpleDNN_result_resized', Pred_result)
