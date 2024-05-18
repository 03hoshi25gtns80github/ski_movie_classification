import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

def add_white_noise(image, noise_level=0.02):
    """画像にホワイトノイズを追加"""
    noise = np.random.randn(*image.size[::-1], 3) * noise_level  # 形状を (高さ, 幅) に変更
    noisy_image = np.array(image) + noise
    return Image.fromarray(np.uint8(np.clip(noisy_image, 0, 255)))

def shift_image(image, shift):
    """画像を左右にシフト"""
    return image.transform(image.size, Image.AFFINE, (1, 0, shift, 0, 1, 0))

def stretch_image(image, rate=1.1):
    """画像を水平方向に引き伸ばす"""
    width, height = image.size
    new_width = int(width * rate)
    return image.resize((new_width, height), Image.BICUBIC)

def augment_image(image_path, output_folder):
    """画像にデータ拡張を適用し、拡張画像を保存"""
    image = Image.open(image_path).convert('RGB')  # RGB画像として読み込み
    base_filename = os.path.splitext(os.path.basename(image_path))[0]

    # ホワイトノイズの追加
    noisy_image = add_white_noise(image)
    noisy_image.save(os.path.join(output_folder, f'{base_filename}_noisy.png'))

    # 画像のシフト
    shifted_image = shift_image(image, shift=10)
    shifted_image.save(os.path.join(output_folder, f'{base_filename}_shifted.png'))

    # 画像のストレッチ
    stretched_image = stretch_image(image, rate=1.1)
    stretched_image.save(os.path.join(output_folder, f'{base_filename}_stretched.png'))

def augment_images_in_folder(mels_folder, output_folder):
    """melsフォルダ内の各サブフォルダにある画像にデータ拡張を適用"""
    for subdir in os.listdir(mels_folder):
        subdir_path = os.path.join(mels_folder, subdir)
        
        if os.path.isdir(subdir_path):
            output_subdir = os.path.join(output_folder, subdir)
            os.makedirs(output_subdir, exist_ok=True)
            
            # サブフォルダ内のすべてのファイルを取得
            for filename in os.listdir(subdir_path):
                if filename.endswith(".png"):
                    file_path = os.path.join(subdir_path, filename)
                    augment_image(file_path, output_subdir)

if __name__ == "__main__":
    mels_folder = "../mels"
    output_folder = "../augment"
    augment_images_in_folder(mels_folder, output_folder)

