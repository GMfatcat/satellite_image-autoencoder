
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 13:54:54 2020

@author: fatcatcat
"""

import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, UpSampling2D, ZeroPadding2D,Input,BatchNormalization, Dropout
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras import initializers
import cv2
#from keras.models import Sequential,Model
from tensorflow.keras.optimizers import SGD, Adam 
from keras.utils import np_utils 
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

f1 = 'D:/building/new_train'
f2 = 'D:/building/new_label'
f3 = 'D:/building/new_test'
f4 = 'D:/building/new_test_label'


print("-----開始讀取數據-----")

#讀取四個不同的照片資料夾
def get_imlist(path):   #此函式讀取特定資料夾下的tif格式影象，返回圖片所在路徑的列表

    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]

imagePath1 = get_imlist(f1)  #r""是防止字串轉譯
imagePath2 = get_imlist(f2)  #r""是防止字串轉譯
imagePath3 = get_imlist(f3)  #r""是防止字串轉譯
imagePath4 = get_imlist(f4)  #r""是防止字串轉譯

#資料夾裡面的照片個數
train_N = len(imagePath1)
trainL_N = len(imagePath2)
test_N = len(imagePath3)
testL_N = len(imagePath4)


#建立四個空陣列
train_data = np.empty((train_N,256,256,3))
train_labels = np.empty((trainL_N,256,256,1))
test_data = np.empty((test_N,256,256,3))
test_labels = np.empty((testL_N,256,256,1))

#將照片轉換成numpy 並儲存進陣列(圖片來源,照片個數,要儲存的陣列)，順便縮小成256*256
#也要將label 照片轉成rgb  跟input 一樣
def loadIMG(imagePath , number, Array):
    while number > 0:
        img = cv2.imread(imagePath[number-1])
        img = cv2.resize(img,(256,256),interpolation=cv2.INTER_AREA)
        img_ndarray=np.asarray(img,dtype='float64')
        Array[number-1] = img_ndarray
        number = number - 1
def loadIMG_gray(imagePath , number, Array):
    while number > 0:
        img = cv2.imread(imagePath[number-1],0)
        img = cv2.resize(img,(256,256),interpolation=cv2.INTER_AREA)
        img_ndarray=np.asarray(img,dtype='float64')
        Array[number-1] = img_ndarray.reshape(256,256,1)
        number = number - 1

loadIMG(imagePath1,train_N,train_data)
loadIMG_gray(imagePath2,trainL_N,train_labels)
loadIMG(imagePath3,test_N,test_data)
loadIMG_gray(imagePath4,testL_N,test_labels)

#檢查是否有匯入
print ("train_data.shape:",train_data.shape)
print(type(train_data))

print ("train_labels.shape:",train_labels.shape)
print(type(train_labels))

print ("test_data.shape:",test_data.shape)
print(type(test_data))

print ("test_labels.shape:",test_labels.shape)
print(type(test_labels))

#正規化到[0,1]之間

train_data = train_data/255
train_labels = train_labels/255
test_data = test_data/255
test_labels = test_labels/255

#開始建立模型
print('-----開始建立模型------')

def train_model():
    
    global history
    
    initializer = tf.keras.initializers.he_normal(seed = None)
    
    input_img= Input(shape=(256, 256, 3))
    # Encoder 使用卷積層，激活函數用 relu，輸入的維度就是上面定義的 input_img
    #大小 = 256*256
    x = Conv2D(16, (3, 3), activation='relu', padding='same',kernel_initializer=initializer)(input_img)
    x = BatchNormalization()(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same',kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same',kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    #大小 = 128*128
    x = Conv2D(32, (3, 3), activation='relu', padding='same',kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same',kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same',kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    #大小 = 64*64
    x = Conv2D(64, (3, 3), activation='relu', padding='same',kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same',kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same',kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    
    #大小 = 32*32
    x = Conv2D(128, (3, 3), activation='relu', padding='same',kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same',kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same',kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same',kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    encoded = MaxPooling2D((2, 2), padding='same', name='encoder')(x)  #大小 = 16*16 為編碼器設置了一個名稱，以便能夠訪問它

    # Decoder 的過程與 Encoder 正好相反，需要跟 Encoder 的神經網絡層做相對應，相對應的激活函數也是一樣，但這邊在解碼中最後一層使用的激活函數是 sigmoid
    x = UpSampling2D((2, 2))(encoded)
    #大小32*32
    x = Conv2D(128, (3, 3), activation='relu', padding='same',kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)    
    x = Conv2D(128, (3, 3), activation='relu', padding='same',kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same',kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same',kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
       
    x = UpSampling2D((2, 2))(x)
    #大小64*64
    x = Conv2D(64, (3, 3), activation='relu', padding='same',kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)    
    x = Conv2D(64, (3, 3), activation='relu', padding='same',kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same',kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    
    x = UpSampling2D((2, 2))(x)
    #大小128*128
    x = Conv2D(32, (3, 3), activation='relu', padding='same',kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)    
    x = Conv2D(32, (3, 3), activation='relu', padding='same',kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same',kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    
    x = UpSampling2D((2, 2))(x)
    #大小256*256
    x = Conv2D(16, (3, 3), activation='relu', padding='same',kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)    
    x = Conv2D(16, (3, 3), activation='relu', padding='same',kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same',kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
        
    # 用 Model 來搭建模型，輸入為圖片，輸出是解碼的結果
    autoencoder = Model(input_img, decoded)  
    
    print(autoencoder.summary())
    
    # 編譯模型，optimizer 使用 adam，loss 使用 categorical_crossentropy 
    autoencoder.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(),metrics=[tf.keras.metrics.MeanSquaredError()])

    history = autoencoder.fit(train_data, train_labels,
                    epochs=200,
                    batch_size=16,
                    shuffle=True,
                    validation_data=(test_data, test_labels))
        
    autoencoder.save('segnet2_outputgray.h5')

print('-----開始訓練模型----')

train_model() 

print('-----訓練完成-------')


#將訓練過程損失可視化
plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['val_mean_squared_error'])
plt.title('Segnet_outputgray_model2 accuracy')
plt.ylabel('error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Segnet_outputgray_Model2 loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# 這邊嘗試 batch = 20 + batch normalization 
  
























