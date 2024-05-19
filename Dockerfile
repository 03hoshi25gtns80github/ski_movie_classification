# ベースイメージとしてPython 3.11を使用
FROM python:3.10 AS base

# NVIDIAのCUDAベースイメージを使用
FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04

# 必要なパッケージをインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Pythonとpipをインストール
COPY --from=base /usr/local/bin/python /usr/local/bin/python
COPY --from=base /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=base /usr/local/lib/libpython3.10.so.1.0 /usr/local/lib/libpython3.10.so.1.0

# LD_LIBRARY_PATHを設定
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# pipを再インストール
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# 作業ディレクトリを設定
WORKDIR /app

# 必要なパッケージをインストール
COPY requirements.txt .
RUN pip install --default-timeout=100 --no-cache-dir -r requirements.txt

# コンテナ内でシェルを起動
CMD ["bash"]