# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 16:31:24 2020

@author: user
"""
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, UpSampling2D, ZeroPadding2D,Input,BatchNormalization, Dropout
from tensorflow.keras.models import Sequential,Model,load_model
import cv2
#from keras.models import Sequential,Model
from tensorflow.keras.optimizers import SGD, Adam 
from keras.utils import np_utils 
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import time


print('----讀取資料----')
t0 = time.time()
#load test image

img = cv2.imread('D:/building/big data/testing/406_90.jpg')
img2 = cv2.imread('D:/building/big data/testing/421_90.jpg')
img3 = cv2.imread('D:/building/big data/testing/428_90.jpg')
img4 = cv2.imread('D:/building/big data/testing/469_90.jpg')
img = img/255
img2 = img2/255
img3 = img3/255
img4 = img4/255

test = np.empty((1,256,256,3),dtype='float32')
test2 = np.empty((1,256,256,3),dtype='float32')
test3= np.empty((1,256,256,3),dtype='float32')
test4 = np.empty((1,256,256,3),dtype='float32')
test[0] = img
test2[0] = img2
test3[0] = img3
test4[0] = img4



print('----讀取完成----')
t1 = time.time()
print('讀取時間:'+ str(round((t1-t0),4))+ ' sec')

print('----開始預測----')
t2 = time.time()
#load model

autoencoder = load_model('segnet2_outputgray_v2.h5')
#autoencoder = load_model('segnet2_outputgray_v2.h5')
result = autoencoder.predict(test)
result2 = autoencoder.predict(test2)
result3 = autoencoder.predict(test3)
result4 = autoencoder.predict(test4)


t3 = time.time()
print('----預測完成----')
print('預測時間:'+ str(round((t3-t2),4)) + ' sec') #小數位四捨五入至第4位 

a = result[0]
b = result2[0]
c = result3[0]
d = result4[0]

r1,th1 = cv2.threshold(a,np.mean(result), 255 , 0)
r2,th2 = cv2.threshold(b,np.mean(result2), 255 , 0)
r3,th3 = cv2.threshold(c,np.mean(result3), 255 , 0)
r4,th4 = cv2.threshold(d,np.mean(result4), 255 , 0)

#load test image ground truth
#ground_truth = cv2.imread('D:/building/big data/not test label/92_original.jpg')
th1 = cv2.cvtColor(th1,cv2.COLOR_GRAY2RGB)
th2 = cv2.cvtColor(th2,cv2.COLOR_GRAY2RGB)
th3 = cv2.cvtColor(th3,cv2.COLOR_GRAY2RGB)
th4 = cv2.cvtColor(th4,cv2.COLOR_GRAY2RGB)
imgstack = np.vstack((img, th1))
imgstack2 = np.vstack((img2, th2))
imgstack3 = np.vstack((img3, th3))
imgstack4 = np.vstack((img4, th4))
imgstack5 = np.hstack((imgstack,imgstack2))
imgstack6 = np.hstack((imgstack3,imgstack4))
imgstack7 = np.hstack((imgstack5,imgstack6))
#imgstack2 = np.hstack((imgstack, ground_truth))

cv2.imshow('original_vs_predict',imgstack7)
cv2.waitKey(0)
'''
cv2.imwrite('D:/building/predict.jpg', th1)
'''



# segnet2_outputgray_v2.h5   是2700張訓練資料的model
# segnet2_outputgray.h5      是1080張訓練資料的model
# 這是使用平均值當作閥值


















