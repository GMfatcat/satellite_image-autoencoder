# satellite_image-autoencoder
專題project  CNN + AUTOENCODER 

#相關資料參考word檔

#segnet3系列為retrain的model

#SEGNET_OUTPUTGRAY = 1080 TRAIN DATA MODEL

#SEGNET_OUTPUTGRAY_V2 = 2700 TRAIN DATA MODEL

#predict / predict_4 is a py file to test model

# small_batch_segnet_all_v2.h5 : this is the fine tune model(學長電腦train的)



#Data Augmentation 
1. 使用 random generate 256x256 py (tif - > jpg)(1 -> 4)

2. increase train & increase test(rotate + flip) (1 -> 6)

3. 增加 24 倍照片

#Use npz file as belows
# NPZ FILE : https://drive.google.com/drive/folders/11Xs5J9A4NZa82fk3_L5TrgM3jSeE6DSM?usp=sharing
內容 : 
train_data.shape: (2700, 256, 256, 3)

<class 'numpy.ndarray'>

train_labels.shape: (2700, 256, 256, 1)

<class 'numpy.ndarray'>

test_data.shape: (168, 256, 256, 3)

<class 'numpy.ndarray'>

test_labels.shape: (168, 256, 256, 1)

<class 'numpy.ndarray'>

引用 :

import numpy as np

file = np.load('位置.../building_data.npz')

train_data = file['train_data']

train_labels = file['train_labels']

test_data = file['test_data']

test_labels = file['test_labels']


#Dhash.py 用於判斷照片相似程度

