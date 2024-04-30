import numpy as np
import time


def sav_jinsi(fi, lamda, dt):
    u = fi.astype(float)
    m, n = u.shape
    uu = np.zeros_like(u)
    flag = 1
    while flag < 100:
        rn = r(fi, u, lamda)                            #标量
        bn = ff_u(fi, u, lamda)/rn                      #矩阵
        cn = u - dt*rn*bn + (dt/2)*np.sum(bn*u)*bn      #矩阵
        gai_bn = AinverseOpt1(m, bn, dt)
        gai_cn = AinverseOpt1(m, cn, dt)
        yn = neiji(bn, gai_bn)
        bn_un1 = neiji(bn, gai_cn)/(1+dt*yn*0.5)      #得到（bn， un+1）
        yidatuo = cn - dt*0.5*bn_un1*bn
        uu = AinverseOpt1(m, yidatuo, dt)
        u = uu
        flag += 1
    return uu


def f_u(fi, u, lamda):
    m, n = u.shape
    tix_u = np.zeros((m, n), dtype=float)
    tiy_u = np.zeros((m, n), dtype=float)
    pro_u = np.pad(u, ((1, 1), (1, 1)), mode='reflect')
    # 计算梯度u
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            tix_u[i - 1, j - 1] = (float(pro_u[i + 1, j]) - float(pro_u[i - 1, j])) / 2
            tiy_u[i - 1, j - 1] = (float(pro_u[i, j + 1]) - float(pro_u[i, j - 1])) / 2
    gen_u = np.sqrt(tix_u**2 + tiy_u**2)
    return lamda*2*(fi-u)**2 - gen_u


def e1_u(fi, u, lamda):
    return np.sum(f_u(fi, u, lamda))


def r(fi, u, lamda):
    return np.sqrt(e1_u(fi, u, lamda))


def ff_u(fi, u, lamda):
    a = tidu_qiudao(u)
    return lamda*(u-fi) + a


def neiji(a, b):
    return np.sum(a*b)


def tidu_qiudao(u):
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
    grad_mag = np.sqrt(tix_u**2 + tiy_u**2)

    # 避免除以零，将等于零的梯度模的值设为一个小正数
    grad_mag[grad_mag == 0] = 1e-6

    # 计算梯度除以梯度的模
    result_x = tix_u/grad_mag   ###注意要是点除
    result_y = tiy_u/grad_mag
    #再扩充
    result_x = np.pad(result_x, ((1, 1), (1, 1)), mode='reflect')
    result_y = np.pad(result_y, ((1, 1), (1, 1)), mode='reflect')
    # 计算结果的散度
    #divergence = cv2.Laplacian(result[0], cv2.CV_64F, ksize=3) + cv2.Laplacian(result[1], cv2.CV_64F, ksize=3)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            divergence[i-1, j-1] += (result_x[i+1, j] - result_x[i-1, j])/2 + (result_y[i, j+1] - result_y[i, j-1])/2
    return divergence


def AinverseOpt1(N, bn, dt):          #这是那个逆算子，N为方图大小
    # solve u - tau*laplace u=f
    rhs_f = np.fft.fft2(bn)
    index = np.concatenate((np.arange(0, N//2), [0], np.arange(-N//2 + 1, 0)))
    sec = -index ** 2  # Don't forget negative sign
    temp = 1 - dt * (np.tile(sec, (N, 1)) + np.tile(sec.reshape(-1, 1), (1, N)))
    res = np.fft.ifft2(rhs_f / temp)
    return np.real(res)
