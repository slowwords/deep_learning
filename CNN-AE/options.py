import argparse

class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--in_dims", type=int, default=3)
        self.parser.add_argument("--latent_dims", type=int, default=256, help="嵌入维度")
        self.parser.add_argument("--device", type=str, default="cpu")
        self.parser.add_argument("--root_path", type=str, default="")
        self.parser.add_argument("--load_size", type=int, default=128)
        self.parser.add_argument("--num_workers", type=int, default=0)
        self.parser.add_argument("--ckpt_path", type=str, default="./checkpoints", help="模型存储路径")
        self.args = self.parser.parse_args()

    @property
    def parse(self):
        return self.args

class TrainOptions(BaseOptions):
    def __init__(self):
        super(TrainOptions, self).__init__()
        self.isTrain = True
        self.parser.add_argument("--train_path", type=str, default="", help="训练集路径")
        self.parser.add_argument("--batch_size", type=int, default=8)
        self.parser.add_argument("--lr", type=float, default=0.0002, help="学习率")
        self.parser.add_argument("--epochs", type=int, default=10, help="总共训练多少轮")
        self.parser.add_argument("--save_epochs", type=int, default=1, help="每多少轮训练存储一次模型")
        self.parser.add_argument("--sample_path", type=str, default="./sample", help="训练样例存储路径")
        self.parser.add_argument("--save_samples", type=int, default=500, help="每多少个batch存储一次样例")
        self.args = self.parser.parse_args()
    @property
    def parse(self):
        return self.args

class TestOptions(BaseOptions):
    def __init__(self):
        super(TestOptions, self).__init__()
        self.isTrain = False
        self.parser.add_argument("--test_path", type=str, default="./test", help="测试集路径")
        self.parser.add_argument("--result_path", type=str, default="./results", help="测试结果存放路径")
        self.parser.add_argument("--batch_size", type=int, default=8)
        self.parser.add_argument("--last_epoch", type=int, default=1, help="测试使用的模型来自哪个epoch")
        self.args = self.parser.parse_args()
    @property
    def parse(self):
        return self.args