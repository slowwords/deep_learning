from torchvision import transforms
import os
from PIL import Image
from torch.utils.data import Dataset

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif', '.TIF', '.tiff', '.TIFF',
]
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    images = sorted(images)
    return images[:min(max_dataset_size, len(images))]

def image_transforms(load_size):   # Resize or CenterCrop
    return transforms.Compose([
        transforms.Resize([load_size, load_size]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

class ImageDataset(Dataset):
    def __init__(self, image_root, load_size, sigma=2., data_mode="centercrop"):
        super(ImageDataset, self).__init__()
        self.image_files = make_dataset(dir=image_root)
        self.number_image = len(self.image_files)
        self.sigma = sigma
        self.load_size = load_size
        self.image_files_transforms = image_transforms(load_size)
    def __getitem__(self, index):
        image = Image.open(self.image_files[index % self.number_image])
        image = self.image_files_transforms(image.convert('RGB'))
        return image
    def __len__(self):
        return self.number_image