import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MelDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(('png', 'jpg', 'jpeg')):
                    self.image_paths.append(os.path.join(subdir, file))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

class SimCLRTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return self.base_transform(x), self.base_transform(x)

class RandomTimeShift:
    def __init__(self, shift_range):
        self.shift_range = shift_range

    def __call__(self, x):
        shift = np.random.randint(-self.shift_range, self.shift_range)
        return np.roll(x, shift, axis=1)

class AddGaussianNoise:
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        noise = np.random.normal(self.mean, self.std, x.shape).astype(np.float32)  # 修正: float32に変換
        return x + noise

def get_data_loader(root_dir, batch_size):
    transform = SimCLRTransform(transforms.Compose([
        transforms.RandomResizedCrop(size=(256, 256)),  # メルスペクトログラムのサイズに合わせて調整
        transforms.Lambda(lambda x: RandomTimeShift(shift_range=10)(np.array(x))),
        transforms.Lambda(lambda x: AddGaussianNoise(mean=0.0, std=0.1)(np.array(x))),
        transforms.ToTensor(),
    ]))
    dataset = MelDataset(root_dir=root_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)  # num_workersを0に設定
    return data_loader