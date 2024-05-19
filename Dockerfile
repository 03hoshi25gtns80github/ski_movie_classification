# ベースイメージとしてPython 3.11を使用
FROM python:3.11 AS base

# NVIDIAのCUDAベースイメージを使用
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Pythonとpipをインストール
COPY --from=base /usr/local/bin/python /usr/local/bin/python
COPY --from=base /usr/local/lib/python3.11 /usr/local/lib/python3.11
RUN apt-get update && apt-get install -y python3-pip

# 作業ディレクトリを設定
WORKDIR /app

# 必要なパッケージをインストール
COPY requirements.txt .
RUN pip3 install --default-timeout=100 --no-cache-dir -r requirements.txt

# コンテナ内でシェルを起動
CMD ["bash"]