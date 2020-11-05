import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, UpSampling2D, ZeroPadding2D,Input,BatchNormalization, Dropout
from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras import initializers
import cv2
#from keras.models import Sequential,Model
from tensorflow.keras.optimizers import SGD, Adam 
from keras.utils import np_utils 
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import time

f1 = 'D:/building/big data/not_train'
f2 = 'D:/building/big data/not_train_label'
f3 = 'D:/building/big data/random_new_rotate_test_2'
f4 = 'D:/building/big data/random_new_rotate_test_label_2'


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


def retrain_model():
    global history
    autoencoder2 = load_model('segnet3.h5')
    autoencoder2.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(),metrics=[tf.keras.metrics.BinaryAccuracy()])

    history = autoencoder2.fit(train_data, train_labels,
                    epochs=150,
                    batch_size=20,
                    shuffle=True,
                    validation_split = 0.15)
        
    autoencoder2.save('segnet3_retrain1.h5')
    



print('-----開始訓練模型----')
t0 = time.time()
retrain_model() 

print('-----訓練完成-------')
t1 = time.time()
print('訓練時間:'+ str(round((t1-t0)/60,2))+ ' min')

#將訓練過程損失可視化
plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('Segnet3_retrain1 accuracy')
plt.ylabel('error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Segnet3_retrain1 loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()












