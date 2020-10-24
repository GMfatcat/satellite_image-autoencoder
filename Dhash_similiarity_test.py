# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 10:56:02 2020

@author: fatcatcat
"""

import cv2

img1 = cv2.imread('4290.jpg',1)
img2 = cv2.imread('42_90.jpg',1)

def dHash(img):
    img=cv2.resize(img,(9,8),interpolation=cv2.INTER_CUBIC)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    dhash_str=''
    
    for i in range(8):                #每行前一個像素大於後一個像素爲1，相反爲0，生成哈希
        for j in range(8):
            if gray[i,j]>gray[i,j+1]:
                dhash_str = dhash_str+'1'
            else:
                dhash_str = dhash_str+'0'
    return dhash_str

def cmpHash(hash1,hash2):              #Hash值對比
    n=0
    if len(hash1)!=len(hash2):         #hash長度不同則返回-1代表傳參出錯
        return -1
    for i in range(len(hash1)):        #遍歷判斷
        if hash1[i]!=hash2[i]:         #不相等則n計數+1，n最終爲相似度
            n=n+1
    return n

hash1 = dHash(img1)
hash2 = dHash(img2)
print(hash1)
print(hash2)

n = cmpHash(hash1,hash2)

if n >10:
    print('n = ' + str(n) + ', it is not similiar')
elif n > 5 and n <= 10:
    print('n = ' + str(n) + ', it is little similiar')
else:
    print('n = ' + str(n) + ', it is  similiar')
    
    



