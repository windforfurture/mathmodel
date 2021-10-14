import  numpy as np
import pandas as pd
from sklearn import metrics
#read_data

class MDS(object):
    def __init__(self):
        pass
    def calculate_distance(self,x, y):
        d = np.sqrt(np.sum((x - y) ** 2))
        return d
    # 计算矩阵各行之间的欧式距离；x矩阵的第i行与y矩阵的第0-j行继续欧式距离计算，构成新矩阵第i行[i0、i1...ij]
    def calculate_distance_matrix(self,x, y):
        d = metrics.pairwise_distances(x, y)
        return d
    def cal_B(sefl,D):
        (n1, n2) = D.shape
        DD = np.square(D)                    # 矩阵D 所有元素平方
        Di = np.sum(DD, axis=1) / n1         # 计算dist(i.)^2
        Dj = np.sum(DD, axis=0) / n1         # 计算dist(.j)^2
        Dij = np.sum(DD) / (n1 ** 2)         # 计算dist(ij)^2
        B = np.zeros((n1, n1))
        for i in range(n1):
            for j in range(n2):
                B[i, j] = (Dij + DD[i, j] - Di[i] - Dj[j]) / (-2)   # 计算b(ij)
        return B
    def MDS(self,data, n=2):
        D = self.calculate_distance_matrix(data, data)
        # print(D)
        B = self.cal_B(D)
        Be, Bv = np.linalg.eigh(B)             # Be矩阵B的特征值，Bv归一化的特征向量
        # print numpy.sum(B-numpy.dot(numpy.dot(Bv,numpy.diag(Be)),Bv.T))
        Be_sort = np.argsort(-Be)
        Be = Be[Be_sort]                          # 特征值从大到小排序
        Bv = Bv[:, Be_sort]                       # 归一化特征向量
        Bez = np.diag(Be[0:n])                 # 前n个特征值对角矩阵
        # print Bez
        Bvz = Bv[:, 0:n]                          # 前n个归一化特征向量
        Z = np.dot(np.sqrt(Bez), Bvz.T).T
        # print(Z)
        return Z








if __name__=='__main__':
    Molecular = pd.read_excel('./data/Molecular_Descriptor.xlsx')
    data=Molecular.values
    data=data[1:-1]
    for i,values in enumerate(data):
        print(len(values))




print("获取到所有的值:\n{}".format(data))



