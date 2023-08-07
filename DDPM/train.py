import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_s_curve
import torch
from ddpm import MLPDiffusion, diffusion_loss_fn, p_sample_loop

class EMA():
    """
    构建一个参数平滑器
    """
    def __init__(self, mu=0.01):
        self.mu = mu
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def __call__(self, name, x):
        assert name in self.shadow
        new_average = self.mu * x + (1.0 - self.mu) * self.shadow[name]
        self.shadow[name] = new_average.clone()
        return new_average

# 计算任意时刻的x的采样值，基于x_0和参数重整化技巧
def q_x(x_0, t):
    """
    可以基于x[0]得到任意时刻t的x[t]
    :param x_0:
    :param t:
    :return:
    """
    noise = torch.randn_like(x_0)
    alphas_t = alphas_bar_sqrt[t]   # t时刻的均值
    alphas_l_m_t = one_minus_alphas_bar_sqrt[t]     # t时刻的标准差
    return (alphas_t * x_0 + alphas_l_m_t * noise)  # 在x[0]的基础上添加噪声

if __name__ == "__main__":

    s_curve, _ = make_s_curve(10**4, noise=0.1)
    s_curve = s_curve[:, [0, 2]] / 10.0

    print("shape of moons: ", np.shape(s_curve))
    data = s_curve.T

    fig, ax = plt.subplots()
    ax.scatter(*data, color='red', edgecolor='white')
    ax.axis('off')

    dataset = torch.Tensor(s_curve).float()

    num_steps = 100

    # 指定每一步的beta
    betas = torch.linspace(-6, 6, num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5   # betas是通过sigmoid计算得到的，所以是递增的，最小值是1e-5，最大值是0.5e-2

    # 计算alpha、alpha_prod、alpha_prod_previous、alpha_bar_sqrt等变量的值
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, 0)  # alpha的连乘
    # alpha_prod_previous从alpha_prod的第一项开始取，把第0项直接令成1
    alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]], 0) # p表示previous
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    assert alphas.shape == alphas_prod.shape == alphas_prod_p.shape == alphas_bar_sqrt.shape \
           == one_minus_alphas_bar_log.shape == one_minus_alphas_bar_sqrt.shape
    print("all the same shape: ", betas.shape)
    # 这些量的形状都取决于num_steps，并且全部一致。但是每一时刻，这些量的值都是不一样的，但是它们都是常数，是不需要训练的，是超参数

    num_shows = 20
    fig, axs = plt.subplots(2, 10, figsize=(28, 3))
    plt.rc('text', color='blue')

    # 共用10000个点，每个点包含两个坐标
    # 生成100步以内，每隔5步加噪后的图像
    for i in range(num_shows):
        j = i // 10
        k = i % 10
        q_i = q_x(dataset, torch.tensor([i*num_steps//num_shows]))  # 生成t时刻的采样数据
        axs[j, k].scatter(q_i[:, 0], q_i[:, 1], color='red', edgecolor='white')
        axs[j, k].set_axis_off()
        axs[j, k].set_title('$q(x_{' + str(i*num_steps//num_shows) + '})$')

    # plt.show()
    plt.savefig("forward.png")

    seed = 1234

    print("Training model...")

    batch_size = 128
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    num_epoch = 4000
    plt.rc('text', color='blue')

    model = MLPDiffusion(num_steps) # 输出维度是2，输入是x和step
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ema = EMA(0.5)

    for t in range(num_epoch):
        for idx, batch_x in enumerate(dataloader):
            loss = diffusion_loss_fn(model, batch_x, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, num_steps)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 1.)   # 对梯度进行clip，保证稳定性
            optimizer.step()
            # for name, param in model.parameters():
                # if param.requires_grad:
                    # param.data = ema(name, param.data)

        # print loss
        if t % 100 == 0:
            print(loss)
            x_seq = p_sample_loop(model, dataset.shape, num_steps, betas, one_minus_alphas_bar_sqrt)    # 共有100个元素

            fig, axs = plt.subplots(1, 10, figsize=(28, 3))
            for i in range(1, 11):
                cur_x = x_seq[i * 10].detach()
                axs[i-1].scatter(cur_x[:, 0], cur_x[:, 1], color='red', edgecolor='white')
                axs[i-1].set_axis_off()
                axs[i-1].set_title('$q(x_{'+str(i*10)+'})$')
            plt.savefig(f"./imgs/epoch_{t}_show.png")
    plt.savefig("./show.png")
    torch.save(model.state_dict(), "./ckpt.pth")