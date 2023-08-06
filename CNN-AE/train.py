import torch
from torch import optim
from net import AE
from options import TrainOptions
from datasets import ImageDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import os
from torchvision import utils

def data_sampler(dataset, shuffle):
    if shuffle:     # 随机抽取样本
        return torch.utils.data.RandomSampler(dataset)
    else:   # 顺序取样
        return torch.utils.data.SequentialSampler(dataset)

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def save_model(opts, net, epoch):   # 保存模型参数
    save_path = f"{opts.ckpt_path}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(
        net.state_dict(),
        f"{opts.ckpt_path}/epoch_{epoch}.pth"
    )

def save_sample(opts, images, epoch, i):    # 训练中保存样例
    assert (isinstance(images, list) or isinstance(images, tuple))
    save_path = f"{opts.sample_path}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if opts.batch_size > 12:
        nrow = opts.batch_size // 2
    else:
        nrow = opts.batch_size
    utils.save_image(
        torch.cat(images, 0),
        f"{opts.sample_path}/epoch-{epoch}-sample-{str(i).zfill(6)}.png",
        nrow=nrow,
        normalize=True,
        range=(-1, 1)
    )

def train(opts, image_data_loader, net, optimizer):
    loader = sample_data(image_data_loader)
    data_loader = iter(loader)
    for epoch in range(opts.epochs):
        epoch += 1
        pbar = tqdm(range(len(image_data_loader)))
        net.train()
        for i in pbar:
            try:
                gt = next(data_loader)
            except (OSError, StopIteration):
                data_loader = iter(loader)
                gt = next(data_loader)
            gt = gt.to(device)
            fake = net(gt)
            # 优化器梯度清零
            optimizer.zero_grad(set_to_none=True)
            # 使用l1距离计算损失
            loss = F.l1_loss(fake, gt).mean()
            # 反向传播损失
            loss.backward()
            # 更新梯度
            optimizer.step()

            state_msg = (
                f"train epoch[{epoch}/{opts.epochs}] l1 loss: {loss:.3f} "
            )
            pbar.set_description(state_msg)

            if (i+1) % opts.save_samples == 0:
                save_sample(opts, [gt, fake], epoch, i+1)

        if epoch % opts.save_epochs == 0:
            save_model(opts, net, epoch)

if __name__ == "__main__":
    opts = TrainOptions().parse
    assert opts.device is not None
    device = torch.device(opts.device)
    net = AE(opts).to(device)
    optimizer = optim.Adam(net.parameters(), lr=opts.lr, betas=(0.9, 0.99))  # 使用Adam优化器
    dataset = ImageDataset(opts.train_path, load_size=opts.load_size)
    image_data_loader = DataLoader(
        dataset,
        batch_size=opts.batch_size,
        sampler=data_sampler(
            dataset, shuffle=True
        ),
        drop_last=True,
        num_workers=opts.num_workers,
        pin_memory=True,
    )
    train(opts, image_data_loader, net, optimizer)