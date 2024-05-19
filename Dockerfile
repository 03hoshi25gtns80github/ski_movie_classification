# ベースイメージとしてPython 3.11を使用
FROM python:3.11

# NVIDIAのCUDAベースイメージを使用
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# 作業ディレクトリを設定
WORKDIR /app

RUN apt-get update && apt-get install -y python3-pip

# 必要なパッケージをインストール
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# コンテナ内でシェルを起動
CMD ["bash"]