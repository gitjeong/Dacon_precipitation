# -*- coding: utf-8 -*-
"""Test of dacon_pcp_ConvLSTM.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1cRzIf3R1YdDc2L6Nfyi1UtOn3wfVWsuy
"""

from google.colab import drive
drive.mount('/content/drive')

import zipfile
path = '/content/drive/My Drive/강우예측AI'
#압축 해제된 파일은 content 즉, colab 상에 위치하므로 세션이 초기화 되면 삭제됨
#압축 해제 위치를 구글 드라이브 강우예측AI 폴더로 지정 가능

zip_file = zipfile.ZipFile(path+'/train.zip')
zip_file.extractall('.')

zip_file = zipfile.ZipFile(path+'/test.zip')
zip_file.extractall('./test/')

# LIBRARY
# glob 모듈의 glob 함수는 사용자가 제시한 조건에 맞는 파일명을 리스트 형식으로 반환한다.

import glob
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, TimeDistributed, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dropout, Dense, Reshape, ConvLSTM2D, GlobalAveragePooling3D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

import warnings
warnings.filterwarnings("ignore")

# train 데이터 생성
"""
trainGenerator()
Each dataset: (120,120,5) array
target: (120,120). dataset의 마지막 부분 -> reshaped to (1,120,120)
remove_minus: target에서 0보다 작은 값은 0으로 바꿔줌
"""

train_files = glob.glob('/content/train/*.npy')

from sklearn.model_selection import train_test_split
train_files, val_files = train_test_split(train_files, test_size = 0.05)
print(len(train_files), len(val_files))

def trainGenerator():
    for file in train_files:
        dataset = np.load(file)
        target= dataset[:,:,-1].reshape(120,120)
        remove_minus = np.where(target < 0, 0, target)
        imsi = dataset[:,:,:4]
        feature = np.moveaxis(imsi, 2, 0).reshape(4,120,120,1)
        yield (feature, remove_minus)

def valGenerator():
    for file in val_files:
        dataset = np.load(file)
        target= dataset[:,:,-1].reshape(120,120)
        remove_minus = np.where(target < 0, 0, target)
        imsi = dataset[:,:,:4]
        feature = np.moveaxis(imsi, 2, 0).reshape(4,120,120,1)
        yield (feature, remove_minus)

train_dataset = tf.data.Dataset.from_generator(trainGenerator, (tf.float32, tf.float32), (tf.TensorShape([4,120,120,1]),tf.TensorShape([120,120])))
val_dataset = tf.data.Dataset.from_generator(valGenerator, (tf.float32, tf.float32), (tf.TensorShape([4,120,120,1]),tf.TensorShape([120,120])))
train_dataset = train_dataset.batch(16, drop_remainder=True).prefetch(1)
val_dataset = val_dataset.batch(16, drop_remainder=True).prefetch(1)

#ConvLSTM

def build_model():
  inputs = Input(shape = (4,120,120,1))
  x = inputs
  x = TimeDistributed(Conv2D(filters=60, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer = tf.keras.regularizers.L1(0.001)))(x)
  x = BatchNormalization()(x)
  x = Dropout(0.5)(x)

  x = TimeDistributed(Conv2D(filters=60, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer = tf.keras.regularizers.L1(0.001)))(x)
  x = BatchNormalization()(x)
  x = Dropout(0.5)(x)

  x = TimeDistributed(Conv2D(filters=60, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer = tf.keras.regularizers.L1(0.001)))(x)
  x = BatchNormalization()(x)
  x = Dropout(0.5)(x)

  x = ConvLSTM2D(filters=30, kernel_size=(3, 3)
                                                , recurrent_activation='hard_sigmoid'
                                                , activation='tanh'
                                                , padding='same', return_sequences=False
                                                , kernel_regularizer = tf.keras.regularizers.L1(0.001)
                                                , dropout=0.5)(x)
  x = BatchNormalization()(x)
  x = Dropout(0.5)(x)
  x = Conv2D(1, kernel_size=(1,1), strides=(1,1), padding='same', activation='relu')(x)
  #x = Flatten()(x)
  x = Reshape((120,120))(x)

  model = Model(inputs = inputs, outputs = x)
  model.compile(loss='mae', optimizer=Adam(learning_rate=0.005), metrics=['mse'])
  return model

# ModelCheckpoint 콜백 & model.fit
best_model = build_model()
best_model.summary()

modelpath = '/content/drive/My Drive/강우예측AI/ConvLSTM_model_1.hdf5'

callbacks = [ModelCheckpoint(filepath = modelpath, monitor = 'val_loss', verbose = 1, save_best_only = True)]
best_model.fit(train_dataset, validation_data = val_dataset, epochs = 30, verbose=1, callbacks = callbacks)

# Test & Submission
#model = tf.keras.models.load_model(path + '/ConvLSTM_model_save')
best_model = tf.keras.models.load_model(modelpath)

test_path = '/content/test'
test_files = sorted(glob.glob(test_path + '/*.npy'))

X_test = []

for file in tqdm(test_files, desc = 'test'):
    data = np.load(file)
    X_test.append(data)

X_test = np.array(X_test)
X_test = np.moveaxis(X_test, 3, 1).reshape(2674,4,120,120,1)
print(X_test.shape)
pred = best_model.predict(X_test)
submission = pd.read_csv('/content/drive/My Drive/강우예측AI/sample_submission.csv')
submission.iloc[:,1:] = pred.reshape(-1, 14400).astype(int)
submission.to_csv(path + '/dacon_pcp_ConvLSTM_submission.csv', index = False)