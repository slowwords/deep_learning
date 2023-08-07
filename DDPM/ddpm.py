import torch
import torch.nn as nn

class MLPDiffusion(nn.Module):
    def __init__(self, n_steps, num_units=128):
        super(MLPDiffusion, self).__init__()

        self.linears = nn.ModuleList(
            [
                nn.Linear(2, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, 2),
            ]
        )

        self.step_embeddings = nn.ModuleList(
            [
                nn.Embedding(n_steps, num_units),
                nn.Embedding(n_steps, num_units),
                nn.Embedding(n_steps, num_units),
            ]
        )

    def forward(self, x_0, t):
        x = x_0
        for idx, embedding_layer in enumerate(self.step_embeddings):
            t_embedding = embedding_layer(t)
            x = self.linears[2*idx](x)
            x += t_embedding
            x = self.linears[2*idx+1](x)
        x = self.linears[-1](x)
        return x

# 损失函数
def diffusion_loss_fn(model, x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps):
    """
    对任意时刻t进行采样计算loss
    :param model:
    :param x_0:
    :param alphas_bar_sqrt:
    :param one_minus_alphas_bar_sqrt:
    :param n_steps:
    :return:
    """
    batch_size = x_0.shape[0]

    # 对一个batchsize样本生成随机时刻t，为了确保t不那么重复，总是先生成一半，另一半用n_steps-1-t，保证随机生成的t能覆盖更多的范围
    t = torch.randint(0, n_steps, size=(batch_size//2,))    # [batch_size]
    t = torch.cat([t, n_steps-1-t], dim=0)  # [batch_size, 1]
    t = t.unsqueeze(-1)
    # print(t.shape)

    # x_0的系数
    a = alphas_bar_sqrt[t]

    # eps的系数
    aml = one_minus_alphas_bar_sqrt[t]

    # 生成随机噪声eps
    e = torch.randn_like(x_0)

    # 构造模型的输入
    x = x_0 * a + e * aml

    # 送入模型，得到t时刻的随机噪声预测值
    output = model(x, t.squeeze(-1))

    # 与真实噪声一起计算误差，求平均值
    return (e - output).square().mean()

def p_sample_loop(model, shape, n_steps, betas, one_minus_alphas_bar_sqrt):
    """
    从x[T]中恢复x[T-1]、x[T-2]、x[T-3]、...、x[0]，多步采样
    :param model:
    :param shape:
    :param n_steps:
    :param betas:
    :param one_minus_alphas_bar_sqrt:
    :return:
    """
    cur_x = torch.randn(shape)
    x_seq = [cur_x]
    for i in reversed(range(n_steps)):
        cur_x = p_sample(model, cur_x, i, betas, one_minus_alphas_bar_sqrt)
        x_seq.append(cur_x)
    return x_seq

def p_sample(model, x, t, betas, one_minus_alphas_bar_sqrt):
    """
    从x[T]采样t时刻的重构值，单步采样
    :param model:
    :param cur_x:
    :param i:
    :param betas:
    :param one_minus_alphas_bar_sqrt:
    :return:
    """
    t = torch.tensor([t])   # 将时刻t转为tensor

    coeff = betas[t] / one_minus_alphas_bar_sqrt[t] # 对应于beta_t / sqrt(1 - alpha_t_bar)

    eps_theta = model(x, t)     # 对应于epsilon_theta(x_t, t)

    mean = (1 / (1-betas[t]).sqrt()) * (x - (coeff * eps_theta))    # 采样计算 mu_theta(x_t, t)

    z = torch.randn_like(x)
    sigma_t = betas[t].sqrt()   # sigma_t是一个确定的值，没有参数

    sample = mean + sigma_t * z  # 采样结果，x_t-1 = mu_t + sigma_t * z

    return (sample)