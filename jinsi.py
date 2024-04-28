import numpy as np
import matplotlib.pyplot as plt
import cv2


def ord_jinsi(fi, lamda, dt):  ###都用浮点数进行计算
    u = fi.astype(float)
    uu = np.zeros_like(u)
    flag = 1
    ru = 0
    while flag < 100:
        la_u = laplacian_five_point(u)                                  #求拉普拉斯u
        uu = u + dt * (la_u - lamda*(fi.astype(float) - u) + tidu(u))   #公式
        # uu[uu > 255] = 255                                            #修正使迭代后取值在0-255之间
        # uu[uu < 0] = 0
        ru = np.linalg.norm(uu-u)/np.linalg.norm(u)                     #计算某个参考指标ru，截断误差
        u[:] = uu    #识别不到
        flag += 1
    return u, ru


def laplacian_five_point(u):
    m, n = u.shape
    la_u = np.zeros((m, n), dtype=float)
    pro_u = np.pad(u, ((1, 1), (1, 1)), mode='reflect')                 #将矩阵扩充一点

    # 计算拉普拉斯，使用周期性边界条件，五点差值
    for i in range(1, m+1):                                             #用五点差分计算拉普拉斯u
        for j in range(1, n+1):
            la_u[i-1, j-1] = (float(pro_u[i + 1, j]) + float(pro_u[i - 1, j]) + float(pro_u[i, j + 1]) + float(pro_u[i, j - 1]) - 4*float(pro_u[i, j]))
    return la_u


def tidu(u):
    # 计算梯度
    # grad_x = cv2.Sobel(u, cv2.CV_64F, 1, 0, ksize=3)
    # grad_y = cv2.Sobel(u, cv2.CV_64F, 0, 1, ksize=3)
    ##初始化
    m, n = u.shape
    tix_u = np.zeros((m, n), dtype=float)
    tiy_u = np.zeros((m, n), dtype=float)
    pro_u = np.pad(u, ((1, 1), (1, 1)), mode='reflect')
    divergence = np.zeros((m, n), dtype=float)

    #计算梯度u
    for i in range(1, m+1):
        for j in range(1, n+1):
            tix_u[i-1, j-1] = (float(pro_u[i + 1, j]) - float(pro_u[i - 1, j]))/2
            tiy_u[i - 1, j - 1] = (float(pro_u[i, j + 1]) - float(pro_u[i, j - 1]))/2

    # 计算梯度的模
    grad_mag = np.sqrt(np.square(tix_u) + np.square(tiy_u))

    # 避免除以零，将等于零的梯度模的值设为一个小正数
    grad_mag[grad_mag == 0] = 1e-6

    # 计算梯度除以梯度的模
    result_x = np.divide(tix_u, grad_mag)   ###注意要是点除
    result_y = np.divide(tiy_u, grad_mag)
    #再扩充
    result_x = np.pad(result_x, ((1, 1), (1, 1)), mode='reflect')
    result_y = np.pad(result_y, ((1, 1), (1, 1)), mode='reflect')
    # 计算结果的散度
    #divergence = cv2.Laplacian(result[0], cv2.CV_64F, ksize=3) + cv2.Laplacian(result[1], cv2.CV_64F, ksize=3)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            divergence[i-1, j-1] += (result_x[i+1, j] - result_x[i-1, j])/2 + (result_y[i, j+1] - result_y[i, j-1])/2
    return divergence
