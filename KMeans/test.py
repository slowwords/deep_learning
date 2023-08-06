import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs     # 从sklearn中直接生成聚类数据
import random
from kmeans import K_Means
import numpy as np

def generate_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    color = "#{:02x}{:02x}{:02x}".format(r, g, b)
    return color

def generate_color_list(num_colors):
    colors = [generate_color() for _ in range(num_colors)]
    return colors

if __name__ == "__main__":
    n_clusters = 5  # 聚类簇的个数
    num_centers = 6
    x, y = make_blobs(n_samples=1000, centers=num_centers, random_state=1234, cluster_std=4)
    # 构造随机样本，其中n_samples为样本点的个数，centers为中心点的个数，cluster_std为聚类的标准差（随机分布的偏差大小）

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.scatter(x[:, 0], x[:, 1], c="red")
    # 创建一个kmeans对象实例
    kmeans = K_Means(n_clusters=n_clusters, max_iter=300)

    # K-Means
    center, labels = kmeans.fit(x)
    colors = generate_color_list(n_clusters)

    plt.subplot(1, 2, 2)
    for point, label in zip(x, labels):
        plt.scatter(point[0], point[1], s=20, color=colors[label])

    # 预测新数据点的类别
    x_new = np.array([[0, 0], [10, 7]])  # 测试用例
    y_pred = kmeans.predict(x_new)
    print(y_pred)
    # 画出预测点的类别分布，预测点以+的形式显示
    for point, label in zip(x_new, y_pred):
        plt.scatter(point[0], point[1], s=400, color=colors[label], marker="+")
    plt.savefig("./show.png")
    plt.show()