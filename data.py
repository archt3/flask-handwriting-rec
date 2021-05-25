
import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
import os
from os import listdir
from os.path import isfile, join
import pandas as pd

def pull_img(folder):
    img_list=[]
    for pic in os.listdir(join('extracted_images',folder)):
        #cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        p=cv2.imread(os.path.join('extracted_images',folder, pic), cv2.IMREAD_GRAYSCALE)
        p=~p
        r, bw_img = cv2.threshold(p,127,255,cv2.THRESH_BINARY)
        c,r=cv2.findContours(bw_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        cnt=sorted(c, key=lambda ctr: cv2.boundingRect(ctr)[0])
        w=28
        h=28
        maxi=0
        for q in cnt:
            x,y,h,w = cv2.boundingRect(q)
            if max(w*h,maxi)==w*h:
                x_m = x
                y_m = y
                w_m = w
                h_m = h
        pic2 = bw_img[y_m:y_m+h_m+10,x_m:x_m+w_m+10]
        pic2 = cv2.resize(pic2,(28,28))
        pic2 = np.reshape(pic2,(784,1))
        img_list.append(pic2)
    return img_list

data=pull_img('-')
for i in range(0,len(data)):
    data[i]=np.append(data[i],['-'])

datap=pull_img('+')
for i in range(0,len(datap)):
    datap[i]=np.append(datap[i],['+'])
datat=pull_img('times')
for i in range(0,len(datat)):
    datat[i]=np.append(datat[i],['*'])

data1=pull_img('1')
for i in range(0,len(data1)):
    data1[i]=np.append(data1[i],['1'])

data2=pull_img('2')
for i in range(0,len(data2)):
    data2[i]=np.append(data2[i],['2'])
data3=pull_img('3')
for i in range(0,len(data3)):
    data3[i]=np.append(data3[i],['3'])
data4=pull_img('4')
for i in range(0,len(data4)):
    data4[i]=np.append(data4[i],['4'])
data5=pull_img('5')
for i in range(0,len(data5)):
    data5[i]=np.append(data5[i],['5'])
data6=pull_img('6')
for i in range(0,len(data6)):
    data6[i]=np.append(data6[i],['6'])
data7=pull_img('7')
for i in range(0,len(data7)):
    data7[i]=np.append(data7[i],['7'])
data8=pull_img('8')
for i in range(0,len(data8)):
    data8[i]=np.append(data8[i],['8'])
data9=pull_img('9')
for i in range(0,len(data9)):
    data9[i]=np.append(data9[i],['9'])

data = np.concatenate((data,datat,datap,data1,data2,data3,data4,data5,data6,data7,data8,data9))
pd.DataFrame(data,index=None).to_csv('processed.csv',index=False)

