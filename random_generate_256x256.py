# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 00:29:25 2020

@author: fatcatcat
"""
import  cv2
import numpy as np
import os
import  random
from datetime import datetime

#建立陣列來儲存隨機生成的座標原點(左上方)



f1 = 'D:/building/test_img'
f2 = 'D:/building/test_label'
f3 = 'D:/building/train_img'
f4 = 'D:/building/train_label'

#讀取四個不同的照片資料夾
def get_imlist(path):   #此函式讀取特定資料夾下的tif格式影象，返回圖片所在路徑的列表

    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.tif')]

imagePath1 = get_imlist(f3)  
imagePath2 = get_imlist(f4)  

train_N = len(imagePath1)
label_N = len(imagePath2)

test1_data = np.empty((train_N,512,512,3))
test2_data = np.empty((label_N,512,512,3))

def loadIMG(imagePath , number, Array):
    while number > 0:
        img = cv2.imread(imagePath[number-1])
        img_ndarray=np.asarray(img,dtype='float64')
        Array[number-1] = img_ndarray
        number = number - 1
  
        
loadIMG(imagePath1,train_N,test1_data)
loadIMG(imagePath2,label_N,test2_data)


Img = np.zeros((4,256,256,3),dtype='float64')

#定義擷取圖片大小 256*256
#目前可以任意產生測試圖片  但是目前無法產生相對應之ground truth

i = 0
k = 1
p1 = './random_test/'
p2 = './random_label/'
p1 = './random_train/'
p2 = './random_train_label/'
def generateIMG(array1,array2,path1,path2):
    global i,k
    for i in range(4):
        random.seed(datetime.now())
        x = random.randint(0,256)
        y = random.randint(0,256)
        point1 = (x,y)
        x2 = random.randint(0,256)
        y2 = random.randint(0,256)
        point2 = (x2,y2)
        x3 = random.randint(0,256)
        y3 = random.randint(0,256)
        point3 = (x3,y3)
        x4 = random.randint(0,256)
        y4 = random.randint(0,256)
        point4 = (x4,y4)
    points = [point1,point2,point3,point4]
    print(points)
    for img in array1:
        while i < 4:
            Img[i] = img[points[i][1]:points[i][1] + 256 , points[i][0]: points[i][0] + 256]
            cv2.imwrite(path1 + str(k) + '_' + str(i)+'.jpg', Img[i])
            i = i + 1 
            
        k = k + 1    
        i = 0
    i = 0
    k = 1
    for img2 in array2:
        while i < 4:
            Img[i] = img2[points[i][1]:points[i][1] + 256 , points[i][0]: points[i][0] + 256]
            cv2.imwrite(path2 + str(k) + '_' + str(i)+'.jpg', Img[i])
            i = i + 1 
            
        k = k + 1
        i = 0
        
generateIMG(test1_data,test2_data,p1,p2)

'''
rotateIMG(test2_data,p2)
'''
