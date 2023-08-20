import numpy as np
from tqdm import tqdm

class model():
    def __init__(self, n_dim, num_steps=100, lr=0.1):
        self.n_dim = n_dim
        self.w = np.zeros(self.n_dim)
        self.b = 0
        self.num_steps = num_steps
        self.lr = lr

    def sigmoid(self, x):
        return 1 / (np.exp(-x) + 1)

    def predictor(self, x):
        """
        x是一个n维向量
        输出预测标量
        """
        z = x.dot(self.w) + self.b
        y_pred = self.sigmoid(z)
        n = y_pred.shape
        """max_index = 0
        for i in range(n[0]):
            if y_pred[i] > y_pred[max_index]:
                max_index = i
        return max_index"""
        return np.round(y_pred)

    def train(self, x, y):
        """
        X:SHAPE=(m,n) np.array, 表示m个训练样本的feature向量
        Y:shape=(m) dumpy.array数组，表示m个训练样本的label，0 or 1
        """
        m, n = x.shape
        for i in tqdm(range(self.num_steps)):
            # 计算预测概率
            z = x.dot(self.w) + self.b
            y_pred = self.sigmoid(z)

            loss = y_pred - y
            print(f"loss: {loss.mean().item():.3f}")
            # 计算梯度
            d_w = loss.dot(x)
            d_b = loss.sum()
            # 更新参数
            self.w -= self.lr * d_w
            self.b -= self.lr * d_b


        # return self.w, self.b

if __name__ == "__main__":
    n_dim = 32
    m = 10
    net = model(n_dim=n_dim, num_steps=m)
    x = np.zeros((m, n_dim), np.float)
    y = np.zeros((m))
    # w, b = net.train(x, y)
    x_test = np.zeros((1, n_dim), np.float)
    y_pred = net.predictor(x_test).item()
    from IPython import embed
    embed()