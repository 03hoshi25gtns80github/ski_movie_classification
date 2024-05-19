# ベースイメージとしてPython 3.11を使用
FROM python:3.11 AS base

# NVIDIAのCUDAベースイメージを使用
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Pythonとpipをインストール
COPY --from=base /usr/local/bin/python /usr/local/bin/python
COPY --from=base /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=base /usr/local/lib/libpython3.11.so.1.0 /usr/local/lib/libpython3.11.so.1.0

# LD_LIBRARY_PATHを設定
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# パッケージリストを更新し、python3-pipをインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリを設定
WORKDIR /app

# 必要なパッケージをインストール
COPY requirements.txt .
RUN pip3 install --default-timeout=100 --no-cache-dir -r requirements.txt

# コンテナ内でシェルを起動
CMD ["bash"]