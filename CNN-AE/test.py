import torch
from torch import optim
from net import AE
from options import TestOptions
from datasets import ImageDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import os
from torchvision import utils

def data_sampler(dataset, shuffle):
    if shuffle:
        return torch.utils.data.RandomSampler(dataset)
    else:
        return torch.utils.data.SequentialSampler(dataset)

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def load_model(opts, net):
    model_dict = torch.load(f"{opts.ckpt_path}/epoch_{opts.last_epoch}.pth")
    net.load_state_dict(model_dict, strict=False)
    print(f"load pretrained AE weights")

def save_result(opts, images, i):
    assert (isinstance(images, list) or isinstance(images, tuple))
    save_path = f"{opts.result_path}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if opts.batch_size > 12:
        nrow = opts.batch_size // 2
    else:
        nrow = opts.batch_size
    utils.save_image(
        torch.cat(images, 0),
        f"{opts.result_path}/result-{str(i).zfill(6)}.png",
        nrow=nrow,
        normalize=True,
        range=(-1, 1)
    )

def test(opts, image_data_loader, net):
    loader = sample_data(image_data_loader)
    data_loader = iter(loader)
    pbar = tqdm(range(len(image_data_loader)))
    load_model(opts, net)
    net.eval()
    for i in pbar:
        try:
            gt = next(data_loader)
        except (OSError, StopIteration):
            data_loader = iter(loader)
            gt = next(data_loader)
        gt = gt.to(device)
        fake = net(gt)

        save_result(opts, [gt, fake], i+1)

if __name__ == "__main__":
    opts = TestOptions().parse
    assert opts.device is not None
    device = torch.device(opts.device)
    net = AE(opts).to(device)
    dataset = ImageDataset(opts.test_path, load_size=opts.load_size)
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
    test(opts, image_data_loader, net)