import  cv2
import numpy as np
import os

f1 = 'D:/building/random_train'
f2 = 'D:/building/random_train_label'

print("-----開始讀取數據-----")

#讀取四個不同的照片資料夾
def get_imlist(path):   #此函式讀取特定資料夾下的tif格式影象，返回圖片所在路徑的列表

    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]

imagePath1 = get_imlist(f1)  #r""是防止字串轉譯
imagePath2 = get_imlist(f2)  #r""是防止字串轉譯

train_N = len(imagePath1)
label_N = len(imagePath2)

test1_data = np.empty((train_N,256,256,3))
test2_data = np.empty((label_N,256,256,3))

def loadIMG(imagePath , number, Array):
    while number > 0:
        img = cv2.imread(imagePath[number-1])
        img = cv2.resize(img,(256,256),interpolation=cv2.INTER_AREA)
        img_ndarray=np.asarray(img,dtype='float64')
        Array[number-1] = img_ndarray
        number = number - 1
  
        
loadIMG(imagePath1,train_N,test1_data)
loadIMG(imagePath2,label_N,test2_data)

k = 0
p1 = './random_new_rotate_train/'
p2 = './random_new_rotate_train_label/'
def rotateIMG(array,path):
    global k
    for img in array:
        #rotate
        cv2.imwrite(path + str(k) + '_original.jpg', img)
        img90=np.rot90(img)
        cv2.imwrite(path + str(k) + '_90.jpg', img90)
        img180=np.rot90(img90)
        cv2.imwrite(path + str(k) +'_180.jpg', img180)
        img270=np.rot90(img180)
        cv2.imwrite(path + str(k) +'_270.jpg', img270)
        #flip 只做左右跟上下 不做左右 + 上下 因為等於選轉180度
        flip1 = cv2.flip(img,1) #左右
        cv2.imwrite(path + str(k) + '_flip1.jpg', flip1)
        flip2 = cv2.flip(img,0) #上下
        cv2.imwrite(path + str(k) + '_flip2.jpg', flip2)
        k = k + 1
        
rotateIMG(test1_data,p1)
k = 0
rotateIMG(test2_data,p2)