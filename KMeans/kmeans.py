import random
import numpy as np
from scipy.spatial.distance import cdist    # 引入scipy中的距离函数，默认欧式距离

class K_Means(object):
    # 初始化，参数n_clusters(即聚成几类，K)、max_iter（迭代次数）、centroids（初始质心）
    def __init__(self, n_clusters=6, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = np.array([])

    # K-Means聚类
    def fit(self, data):
        """
        :param data: 数据
        :return: 聚类中心列表和类别索引列表
        """
        # 假如没有指定初始质心，就随机选取data中的点作为初始质心
        if self.centroids.shape == (0,):
            # 从data中随机生成0-data行数的k个整数作为索引值
            self.centroids = data[np.random.randint(0, data.shape[0], self.n_clusters), :]  # data.shape[0]为data行数，生成self.n_clusters个即6个

        # 开始迭代
        for i in range(self.max_iter):
            # 1.计算距离矩阵，得到的是一个100*6的矩阵（每一行代表一个样本点到所有质心的距离，一行里六个值分别指到第几个质心的距离）
            distances = cdist(data, self.centroids)  # cdist()只要要求同维度就可以

            # 2.对距离按由近到远排序，选取最近的质心点的类别作为当前点的分类
            c_index = np.argmin(distances, axis=1)  # axis=1每一行取最小值，最后结果保存为一列（100*1的矩阵）

            # 3.对每一类数据进行均值计算，更新质心点坐标
            for i in range(self.n_clusters):
                # 首先排除掉没有出现在c_index里的类别（即所有的点都没有离这个质心最近）
                if i in c_index:  # i为0-5
                    # 选出所有类别是i的点，取data里面坐标的均值，更新第i个质心
                    # c_index==i逻辑判断表达式，结果为布尔类型（数组）。c_index为一个数组，data[c_index==i]返回结果为true对应的data的值，即类别为i的值的坐标
                    self.centroids[i] = np.mean(data[c_index == i], axis=0)  # 布尔索引，axis=0得到一行的数据，将每一列做均值计算，列数不变
        return self.centroids, c_index

    # 实现预测方法
    def predict(self, samples):  # samples一组样本点（新来的测试数据）
        """
        :param samples: 待预测样本
        :return: 类别索引
        """
        # 跟上面一样，先计算距离矩阵，然后选取距离最近的那个质心的类别
        distances = cdist(samples, self.centroids)
        c_index = np.argmin(distances, axis=1)
        return c_index

