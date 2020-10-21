# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 12:15:12 2020

@author: fatcatcat
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

img = cv2.imread('D:/building/big data/not test/92_original.jpg')
img_2 = img
img = img/255

test = np.empty((1,256,256,3))
test[0] = img 



print('----讀取完成----')
t1 = time.time()
print('讀取時間:'+ str(round((t1-t0),4))+ ' sec')

print('----開始預測----')
t2 = time.time()
#load model
autoencoder = load_model('segnet2_outputgray_v2.h5')
#autoencoder = load_model('segnet2_outputgray_v2.h5')
result = autoencoder.predict(test)
t3 = time.time()
print('----預測完成----')
print('預測時間:'+ str(round((t3-t2),4)) + ' sec') #小數位四捨五入至第4位 
#像素位置x y 
a=result[0]
ret, th1 = cv2.threshold(a, np.mean(result), 255 , 0) # 如果大於平均值np.mean(result)//目前先用255/22當閾值  則填充灰階255 ，相反則0
'''
#設立灰階thershold 
for x in range(256) :
    
    for y in range(256):
        # [b,g,r] /3 看有沒有大於平均值  np.mean(result)
        if np.mean(a[x,y]) > np.mean(result):
            
            a[x,y] = 255
            
        else:
            
            a[x,y] = 0
'''


#print(result)
'''
cv2.imshow('original',img_2)
cv2.waitKey(0)

cv2.imshow('predict',th1)
cv2.waitKey(0)
'''
#load test image ground truth
ground_truth = cv2.imread('D:/building/big data/not test label/92_original.jpg')
th1 = cv2.cvtColor(th1,cv2.COLOR_GRAY2RGB)
imgstack = np.hstack((img, th1))
#imgstack2 = np.hstack((imgstack, ground_truth))
cv2.imshow('original_vs_predict',imgstack)
cv2.waitKey(0)
#cv2.imshow('original_vs_predict_vs_ground_truth',imgstack)
#cv2.waitKey(0)
'''
cv2.imwrite('D:/building/predict.jpg', th1)
'''



# segnet2_outputgray_v2.h5   是2700張訓練資料的model
# segnet2_outputgray.h5      是1080張訓練資料的model














