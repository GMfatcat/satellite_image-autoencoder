# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 16:22:12 2020

@author: user
"""


import cv2
import numpy as np

a = cv2.imread('D:/building/big data/testing/469_90.jpg',0)
cv2.imshow('original',a)
cv2.waitKey(0)

def otsu(gray):
    pixel_number = gray.shape[0] * gray.shape[1]
    mean_weigth = 1.0/pixel_number
    his, bins = np.histogram(gray, np.array(range(0, 256)))
    final_thresh = -1
    final_value = -1
    for t in bins[1:-1]: # This goes from 1 to 254 uint8 range (Pretty sure wont be those values)
        Wb = np.sum(his[:t]) * mean_weigth
        Wf = np.sum(his[t:]) * mean_weigth

        mub = np.mean(his[:t])
        muf = np.mean(his[t:])

        value = Wb * Wf * (mub - muf) ** 2

        print("Wb", Wb, "Wf", Wf)
        print("t", t, "value", value)

        if value > final_value:
            final_thresh = t
            final_value = value
    final_img = gray.copy()
    print(final_thresh)
    final_img[gray > final_thresh] = 255
    final_img[gray < final_thresh] = 0
    return final_img


print('finish')
cv2.imshow('otzu',otsu(a))
cv2.waitKey(0)
