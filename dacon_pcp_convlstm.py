# -*- coding: utf-8 -*-
"""dacon_pcp_ConvLSTM.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ENr2Zlu-68V0FTpJyi0LAiMUTuAt70Px
"""

#from google.colab import drive
#drive.mount('/content/drive')

"""
import zipfile
path = '/content/drive/My Drive/강우예측AI'
#압축 해제된 파일은 content 즉, colab 상에 위치하므로 세션이 초기화 되면 삭제됨
#압축 해제 위치를 구글 드라이브 강우예측AI 폴더로 지정 가능

zip_file = zipfile.ZipFile(path+'/train.zip')
zip_file.extractall('.')

zip_file = zipfile.ZipFile(path+'/test.zip')
zip_file.extractall('./test/')
"""

# LIBRARY
# glob 모듈의 glob 함수는 사용자가 제시한 조건에 맞는 파일명을 리스트 형식으로 반환한다.

import glob
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import ConvLSTM2D, Conv2D, Conv3D, BatchNormalization, Input
from tensorflow.keras import Model

import warnings
warnings.filterwarnings("ignore")

# train 데이터 생성
"""
trainGenerator()
Each dataset: (120,120,5) array
target: (120,120). dataset의 마지막 부분 -> reshaped to (1,1,120,120)
remove_minus: target에서 0보다 작은 값은 0으로 바꿔줌
"""

train_files = glob.glob('/home/ML_Projects/dacon_pcp/data/train/*.npy')
print(len(train_files))

def trainGenerator():
    for file in train_files:
        dataset = np.load(file)
        target= dataset[:,:,-1].reshape(1, 120, 120)
        remove_minus = np.where(target < 0, 0, target)
        feature = dataset[:,:,:4].reshape(4, 1, 120, 120)

        yield (feature, remove_minus)
        
train_dataset = tf.data.Dataset.from_generator(trainGenerator, (tf.float32, tf.float32), (tf.TensorShape([4,1,120,120]),tf.TensorShape([1,120,120])))
train_dataset = train_dataset.batch(256).prefetch(1)

#ConvLSTM
"""
참고자료
1. An introduction to ConvLSTM
https://medium.com/neuronio/an-introduction-to-convlstm-55c9025563a7
2. Next-frame prediction with Conv-LSTM
https://keras.io/examples/vision/conv_lstm/

"""

def ConvLSTM_model(input_layer):

  ConvLSTM_1 = ConvLSTM2D(filters=40, kernel_size=(3, 3)
                                                , recurrent_activation='hard_sigmoid'
                                                , activation='tanh'
                                                , padding='same', return_sequences=True
                                                , data_format='channels_first')(input_layer)
  BatchNormalization_1 = BatchNormalization()(ConvLSTM_1)

  ConvLSTM_2 = ConvLSTM2D(filters=40, kernel_size=(3, 3)
                                                , recurrent_activation='hard_sigmoid'
                                                , activation='tanh'
                                                , padding='same', return_sequences=False
                                                , data_format='channels_first')(BatchNormalization_1)
  BatchNormalization_2 = BatchNormalization()(ConvLSTM_2)
  
  output_layer = Conv2D(filters=1, kernel_size=(1,1), padding="same", activation='relu', data_format='channels_first')(BatchNormalization_2)

  return output_layer

input_layer = Input((4, 1, 120, 120))
output_layer = ConvLSTM_model(input_layer)

model = Model(input_layer, output_layer)

model.summary()

model.compile(loss='mae', optimizer='adam')
model.fit(train_dataset, epochs = 5, verbose=1)

