import os
import torch
import pandas as pd
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from torchvision import transforms
from train import get_model
from dataset import MelSpectrogramDataset
from utils import convert_mts_to_wav, convert_audio_to_melspectrogram
from pyclustering.cluster.xmeans import xmeans, kmeans_plusplus_initializer
from pyclustering.cluster.center_initializer import random_center_initializer

# エンコーダの読み込み
save_dir = '../SimCLR'
encoder_path = os.path.join(save_dir, 'SimCLR_encoder_bs32_od128_temp0.5_lr0.0003_epoch4.pth')
model = get_model(out_dim=128)
model.load_state_dict(torch.load(encoder_path))
model.eval()

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
        
        # メタデータに追加
        metadata.append({'original_filename': filename})

# メタデータをDataFrameに変換
df = pd.DataFrame(metadata)

# データローダーの作成
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
dataset = MelSpectrogramDataset(mels_folder, transform=transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False)

# データのエンコード
embeddings = []
filenames = []  # ファイル名を保存するリスト
with torch.no_grad():
    for (x_i, paths) in data_loader:  # データローダーから画像パスも取得
        z_i = model(x_i)
        embeddings.append(z_i)
        filenames.extend([os.path.basename(path) for path in paths])  # ファイル名をリストに追加

embeddings = torch.cat(embeddings).numpy()

# X-meansクラスタリング
# ハイパーパラメータの設定
min_clusters = 2
max_clusters = 8

# 初期クラスタの設定
initial_centers = kmeans_plusplus_initializer(embeddings, min_clusters).initialize()
xmeans_instance = xmeans(embeddings, initial_centers, max_clusters)
xmeans_instance.process()
clusters = xmeans_instance.get_clusters()
labels = [0] * len(embeddings)

for cluster_id, cluster in enumerate(clusters):
    for index in cluster:
        labels[index] = cluster_id

# クラスタリング結果をメタデータに追加
df['cluster'] = 0  # 初期化
for filename, label in zip(filenames, labels):
    df.loc[df['original_filename'] == filename, 'cluster'] = label

# データフレームをクラスタ順に並び替え
df = df.sort_values(by='cluster')
# メタデータを表示
print("Metadata DataFrame with Clustering Results:")
print(df)

# メタデータをCSVファイルとして保存
df.to_csv('metadata_with_clusters.csv', index=False)
print("Metadata with clustering results saved to metadata_with_clusters.csv")