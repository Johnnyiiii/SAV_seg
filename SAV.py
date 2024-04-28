import numpy as np


def sav_jinsi(lamda, dt, fi):
    u = fi.astype(float)
    uu = np.zeros_like(u)
    flag = 1
    while flag < 100:
        rn = r(fi, u, lamda)                   #biaoliang
        bn = ff_u(fi, u, lamda)/rn             #jvzhen
        cn = u - dt*rn*bn + (dt/2)*np.sum(bn*u)*bn
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
    return 1


def neiji(a, b):
    return np.sum(a*b)
