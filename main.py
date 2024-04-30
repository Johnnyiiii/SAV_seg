import matplotlib.pyplot as plt
import numpy as np
import cv2
import time

import jinsi
from pic import Picture
import k_mean
import SAV

K = 3
lamda = 1.5
Ru = [1, 1, 1, 1]

time1 = time.time()

pic = Picture("C:/Users/Administrator/Desktop/bishe/tu/niao.jpg")
pic.fenlei()
time2 = time.time()
pic.gray2d()
time3 = time.time()
pic.gray2IIH()
#这三步是生成iih图像
time4 = time.time()
if pic.tongdaoshu == 3:
    #u = [1-pic.Lab[:, :, 0]/255, 1-pic.Lab[:, :, 1]/255, 1-pic.Lab[:, :, 2]/255, pic.IIH]  ###改成0到1之间
    u = [1-pic.tongdao[:, :, 0] / 255, 1-pic.tongdao[:, :, 1] / 255, 1-pic.tongdao[:, :, 2] / 255, pic.IIH]    #RGB做的
    uu = np.zeros_like(u)
    # for i in range(4): p
    #     uu[i], Ru[i] = jinsi.ord_jinsi(u[i], lamda=1.5, dt=0.003) #调整lambda,时间步长dt和迭代次数
    time5 = time.time()
    #这里是计算平滑近似
    uu[0], Ru[0] = SAV.sav_jinsi(u[0], lamda=1.5, dt=0.005)
    time6 = time.time()
    uu[1], Ru[1] = SAV.sav_jinsi(u[1], lamda=1.5, dt=0.002)
    time7 = time.time()
    uu[2], Ru[2] = SAV.sav_jinsi(u[2], lamda=1.5, dt=0.002)
    time8 = time.time()
    uu[3], Ru[3] = SAV.sav_jinsi(u[3], lamda=1.5, dt=0.006)
    time9 = time.time()
    #uu = uu.astype(dtype='uint8')

    #这里打印八幅图，上面是近似前，下面是近似后
    # plt.subplot(4, 2, 1)
    # plt.imshow(u[0]*255, cmap=plt.cm.binary, vmin=0, vmax=255)
    # plt.subplot(4, 2, 2)
    # plt.imshow(u[1]*255, cmap=plt.cm.binary, vmin=0, vmax=255)
    # plt.subplot(4, 2, 3)
    # plt.imshow(u[2]*255, cmap=plt.cm.binary, vmin=0, vmax=255)
    # plt.subplot(4, 2, 4)
    # plt.imshow(u[3]*255, cmap=plt.cm.binary, vmin=0, vmax=255)
    # plt.subplot(4, 2, 5)
    # plt.imshow(uu[0]*255, cmap=plt.cm.binary, vmin=0, vmax=255)
    # plt.subplot(4, 2, 6)
    # plt.imshow(uu[1]*255, cmap=plt.cm.binary, vmin=0, vmax=255)
    # plt.subplot(4, 2, 7)
    # plt.imshow(uu[2]*255, cmap=plt.cm.binary, vmin=0, vmax=255)
    # plt.subplot(4, 2, 8)
    # plt.imshow(uu[3]*255, cmap=plt.cm.binary, vmin=0, vmax=255)
    # plt.show()
    # print(Ru)

    # ku = np.zeros_like(uu)
    ku = uu
    # zk = np.zeros_like(uu[0])       #zk表示每个像素点属于哪个簇
    r1 = np.random.random(4)
    r2 = np.random.random(4)          #k改变
    r3 = np.random.random(4)
    m, n = ku[0].shape
    zk = np.zeros((m, n))
    for i in range(3):
        zk, s1, s2, s3 = k_mean.kmeans_main(ku, 2, r1, r2, r3)    #k表示分为几簇，这里为2不变
        #zk, s1, s2 = k_mean.kmeans_main(ku, 2, r1, r2)
        r1, r2, r3 = s1, s2, s3
        #r1, r2 = s1, s2
    time10 = time.time()
    for i in range(m):
        for j in range(n):
            if zk[i, j] == 0:
                ku[:, i, j] = r1
            if zk[i, j] == 1:
                ku[:, i, j] = r2
            if zk[i, j] == 2:
                ku[:, i, j] = r3
    time11 = time.time()
    #这里打印处理前与结果图
    # plt.subplot(4, 2, 1)
    # plt.imshow(pic.Lab[:, :, 0], cmap=plt.cm.binary, vmin=0, vmax=255)
    # plt.subplot(4, 2, 2)
    # plt.imshow(pic.Lab[:, :, 1], cmap=plt.cm.binary, vmin=0, vmax=255)
    # plt.subplot(4, 2, 3)
    # plt.imshow(pic.Lab[:, :, 2], cmap=plt.cm.binary, vmin=0, vmax=255)
    # plt.subplot(4, 2, 4)
    # plt.imshow(pic.IIH*225, cmap=plt.cm.binary, vmin=0, vmax=255)
    # plt.subplot(4, 2, 5)
    # plt.imshow(ku[0]*255, cmap=plt.cm.binary, vmin=0, vmax=255)
    # plt.subplot(4, 2, 6)
    # plt.imshow(ku[1]*255, cmap=plt.cm.binary, vmin=0, vmax=255)
    # plt.subplot(4, 2, 7)
    # plt.imshow(ku[2]*255, cmap=plt.cm.binary, vmin=0, vmax=255)
    # plt.subplot(4, 2, 8)
    # plt.imshow(ku[3]*255, cmap=plt.cm.binary, vmin=0, vmax=255)
    # plt.show()
    #或者是打印最后的结果
    plt.imshow((ku[0] * 255 + ku[1] * 255 + ku[2] * 255 + ku[3] * 255) / 4, cmap=plt.cm.binary, vmin=0, vmax=255)
    time12 = time.time()
    plt.show()
    print(time2-time1, time3-time2, time4-time3, time5-time4, time6-time5, time7-time6, time8-time7, time9-time8, time10-time9, time11-time10, time12-time11)
elif pic.tongdaoshu == 1:            ##决定只做彩色图像了，以下废弃
    u = [pic.gray, pic.IIH]
    uu = np.zeros_like(u)
    for i in range(2):
        uu[i], Ru[i] = jinsi.ord_jinsi(u[i], lamda=1.5, dt=0.01)
    plt.imshow(pic.IIH, cmap=plt.cm.binary, vmin=0, vmax=255)
    plt.show()
