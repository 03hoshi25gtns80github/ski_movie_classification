import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class MelSpectrogramDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        # ラベルを数値にマッピングする辞書を作成（アルファベット順にソート）
        self.label_map = {label: idx for idx, label in enumerate(sorted(self.annotations['label'].unique()))}
        
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        label_str = self.annotations.iloc[idx, 1]  # ラベルを文字列として取得
        label = self.label_map[label_str]  # ラベルを数値に変換

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label)  # ラベルをTensorに変換