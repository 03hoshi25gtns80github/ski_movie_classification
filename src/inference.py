import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from model import create_model
from get_mels import convert_mts_to_wav, convert_audio_to_melspectrogram  # get_mels.pyの関数をインポート
import pandas as pd

# モデルの読み込み
model_path = "../model/trained_model.pth"
num_classes = 4  # クラス数を適切に設定
model = create_model(num_classes=num_classes)
model.load_state_dict(torch.load(model_path))
model.eval()

# データ変換の定義
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# メタデータの作成
test_folder = "../test"
wav_folder = "../test/wav"
mels_folder = "../test/mels"
metadata = []

# フォルダの作成
os.makedirs(wav_folder, exist_ok=True)
os.makedirs(mels_folder, exist_ok=True)

# 動画ファイルから音声ファイルを作成
convert_mts_to_wav(test_folder, wav_folder)

# 音声ファイルからメルスペクトログラムを作成
convert_audio_to_melspectrogram(wav_folder, mels_folder)

# メルスペクトログラム画像のメタデータを作成
for filename in os.listdir(mels_folder):
    if filename.endswith("_melspectrogram.png"):
        label = filename.split('-')[0]
        mel_image_path = os.path.join(mels_folder, filename)
        
        # メタデータに追加
        metadata.append((mel_image_path, label))

# メタデータをDataFrameに変換
df = pd.DataFrame(metadata, columns=['image_path', 'label'])

# ラベルを数値にマッピングする辞書を作成（アルファベット順にソート）
label_map = {label: idx for idx, label in enumerate(sorted(df['label'].unique()))}

# DataFrameを表示
print("Metadata DataFrame:")
print(df)

# 推論と正解率の計算
correct = 0
total = 0

for index, row in df.iterrows():
    image_path = row['image_path']
    true_label_str = row['label']
    true_label = label_map[true_label_str]  # ラベルを数値に変換
    
    # 画像の読み込みと変換
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # バッチサイズの次元を追加
    
    # 推論
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
    
    # 正解ラベルと比較
    if predicted.item() == true_label:
        correct += 1
    total += 1

accuracy = 100 * correct / total
print(f'Accuracy: {accuracy:.2f}%')

# 正解率をテキストファイルに保存
accuracy_file_path = os.path.join(os.path.dirname(model_path), 'accuracy.txt')
with open(accuracy_file_path, 'w') as f:
    f.write(f'Accuracy: {accuracy:.2f}%\n')
print(f'Accuracy: {accuracy:.2f}%')
