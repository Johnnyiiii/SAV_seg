#需要迭代几次？
#簇如何取？
import numpy as np
import cv2
import random


def fanshu2(a, b):                #计算两个四维向量之间的距离
    return np.sqrt(np.sum((a - b) ** 2))


def kmeans_main(uu, k, r1, r2, r3):    #输入图像与三个平均向量
#def kmeans_main(uu, k, r1, r2):
    if k == 3:
        flag1, flag2, flag3 = 0, 0, 0            #标记每个簇像素点个数
        ji1, ji2, ji3 = 0, 0, 0                  #计算每个簇积分
        m, n = uu[0].shape
        d = np.zeros((m, n))
        for i in range(m):
            for j in range(n):        #遍历元素
                l = uu[:, i, j]
                d[i, j] = np.argmin([fanshu2(l, r1), fanshu2(l, r2), fanshu2(l, r3)])
        for i in range(m):
            for j in range(n):
                if d[i, j] == 0:
                    flag1 += 1
                    ji1 += uu[0:4, i, j]
                if d[i, j] == 1:
                    flag2 += 1
                    ji2 += uu[0:4, i, j]
                if d[i, j] == 2:
                    flag3 += 1
                    ji3 += uu[0:4, i, j]
        # 计算新的平均向量
        if flag1 > 0:
            s1 = ji1/flag1
        else:
            s1 = np.random.random(4)
        if flag2 > 0:
            s2 = ji2 / flag2
        else:
            s2 = np.random.random(4)
        if flag3 > 0:
            s3 = ji3 / flag3
        else:
            s3 = np.random.random(4)
    elif k == 2:
        flag1, flag2 = 0, 0
        ji1, ji2 = 0, 0
        m, n = uu[0].shape
        d = np.zeros((m, n))
        for i in range(m):
            for j in range(n):  # 遍历元素
                l = uu[:, i, j]
                d[i, j] = np.argmin([fanshu2(l, r1), fanshu2(l, r2)])
        for i in range(m):
            for j in range(n):
                if d[i, j] == 0:
                    flag1 += 1
                    ji1 += uu[0:4, i, j]
                if d[i, j] == 1:
                    flag2 += 1
                    ji2 += uu[0:4, i, j]
        # 计算新的平均向量
        if flag1 > 0:
            s1 = ji1 / flag1
        else:
            s1 = np.random.random(4)
        if flag2 > 0:
            s2 = ji2 / flag2
        else:
            s2 = np.random.random(4)
        s3 = 1
    else:
        print('fault')
        d, s1, s2, s3 = 0, 0, 0, 0
    return d, s1, s2, s3             #输出每个像素点属于哪个簇，以及新的平均向量


# v1 = np.random.rand(4, 3, 2)
# print(v1[0].shape)
# m, n = v1[0].shape
# print(m, n)
# v1[:, 0, 1] = [1, 2, 3, 4]
# print(v1)

# v2 = v1[0:4, 2, 1]
# print(v2, np.argmin(v2))
# print(np.sqrt(np.sum((v1 - v2) ** 2)))

# r1, r2, r3 = 0, 1, 3
# print(r1, r2, r3)
# s1, s2, s3 = r1+2, r2-1, r3+4
# print(s1, s2, s3)
# r1, r2, r3 = s1, s2, s3
# print(s1, s2, s3, r1, r2, r3)
