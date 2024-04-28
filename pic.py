import numpy as np
import matplotlib.pyplot as plt
import cv2


class Picture:

    def __init__(self, path):
        self.path = path
        self.tongdao = plt.imread(self.path)   #读取全部通道
        self.tongdaoshu = 3
        self.gray = (0, 0)
        self.Lab = 0
        self.d = 0
        self.average = np.zeros((self.tongdao.shape[0], self.tongdao.shape[1]))
        self.seta_x = np.zeros((self.tongdao.shape[0], self.tongdao.shape[1]))
        self.chaf = np.zeros((self.tongdao.shape[0], self.tongdao.shape[1]))
        self.IIH = np.zeros((self.tongdao.shape[0], self.tongdao.shape[1]))

    def fenlei(self):                          #看图片是单通道还是多通道
        # if self.tongdao.shape[2] == 4:
        #     self.tongdao = self.tongdao[:, :, :3]
        if self.tongdao.shape[2] == 1:
            self.tongdaoshu = 1
            self.gray = self.tongdao
        elif self.tongdao.shape[2] == 3:
            self.tongdaoshu = 3
            self.Lab = cv2.cvtColor(self.tongdao, cv2.COLOR_RGB2Lab)
            self.gray = cv2.cvtColor(self.tongdao, cv2.COLOR_RGB2GRAY)
        else:
            print("通道数应该为一或三")

    def gray2d(self):                            #计算d的值
        for xj in range(self.tongdao.shape[0]):             #拆成了很多的部分来计算
            for xi in range(self.tongdao.shape[1]):
                for yj in range(xj-1, xj+2):
                    for yi in range(xi-1, xi+2):
                        if 0 <= yj < self.tongdao.shape[0] and 0 <= yi < self.tongdao.shape[1]:
                            self.average[xj][xi] += self.gray[yj][yi]
                            self.seta_x[xj][xi] += 1
                self.average[xj][xi] *= 1/self.seta_x[xj][xi]
                for yj in range(xj-1, xj+2):
                    for yi in range(xi-1, xi+2):
                        if 0 <= yj < self.tongdao.shape[0] and 0 <= yi < self.tongdao.shape[1]:
                            self.chaf[xj][xi] += pow(self.gray[yj][yi]-self.average[xj][xi] if self.gray[yj][yi] >= self.average[xj][xi] else self.average[xj][xi]-self.gray[yj][yi], 2)
                self.d += self.chaf[xj][xi]/self.seta_x[xj][xi]
        self.d *= 1/(self.tongdao.shape[0] * self.tongdao.shape[1])

    def X_y(self, xi, xj, yi, yj):
        if pow(self.gray[yj][yi]-self.gray[xj][xi] if self.gray[yj][yi] >= self.gray[xj][xi] else self.gray[xj][xi]-self.gray[yj][yi], 2) >= self.d:
            return 0
        elif pow(self.gray[yj][yi]-self.gray[xj][xi] if self.gray[yj][yi] >= self.gray[xj][xi] else self.gray[xj][xi]-self.gray[yj][yi], 2) < self.d:
            return 1

    def gray2IIH(self):
        for xj in range(self.tongdao.shape[0]):
            for xi in range(self.tongdao.shape[1]):
                for yj in range(xj - 1, xj + 2):
                    for yi in range(xi - 1, xi + 2):
                        if 0 <= yj < self.tongdao.shape[0] and 0 <= yi < self.tongdao.shape[1]:
                            self.IIH[xj][xi] += self.X_y(xi, xj, yi, yj)
                self.IIH[xj][xi] *= 1/self.seta_x[xj][xi]






