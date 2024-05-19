"""
    movieフォルダ内の映像に対するメルスペクトラム画像を作成
    処理後のデータ整理は手動で行う
"""

import os
import shutil
import librosa
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from PIL import Image


def convert_mts_to_wav(input_folder, output_folder):
    # 入力フォルダ内のすべてのファイルを取得
    for filename in os.listdir(input_folder):
        if filename.endswith((".MTS", ".mp4")):
            input_path = os.path.join(input_folder, filename)
            output_filename = os.path.splitext(filename)[0] + ".wav"
            output_path = os.path.join(output_folder, output_filename)
            
            # 動画ファイルを読み込み
            video = VideoFileClip(input_path)
            
            # 動画の長さを確認
            if video.duration < 5:
                print(f"Skipped {input_path} because it is shorter than 5 seconds")
                continue
            
            # 最初の5秒の音声を抽出
            audio = video.audio.subclip(0, 5)
            
            # 音声を.wavファイルとして保存
            audio.write_audiofile(output_path, codec='pcm_s16le')
            print(f"Converted {input_path} to {output_path}")
            
def convert_audio_to_melspectrogram(input_folder, output_folder):
    # 入力フォルダ内のすべてのファイルを取得
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            input_path = os.path.join(input_folder, filename)
            output_filename = os.path.splitext(filename)[0] + "_melspectrogram.png"
            output_path = os.path.join(output_folder, output_filename)
            
            # 音声ファイルを読み込み
            y, sr = librosa.load(input_path, sr=None)
            
            # メルスペクトログラムを計算
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            S_dB = librosa.power_to_db(S, ref=np.max)
            
            # メルスペクトログラムを画像として保存
            plt.figure(figsize=(10, 4))
            plt.axis('off')  # 軸を非表示にする
            librosa.display.specshow(S_dB, sr=sr, x_axis=None, y_axis=None, cmap='viridis')
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()

            # 画像をRGBに変換して再保存
            image = Image.open(output_path).convert('RGB')
            image.save(output_path)
            print(f"Converted {input_path} to {output_path}")
            
def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def main():
    movie_folder = "../movie"
    wav_folder = "../wav"
    mels_folder = "../mels"

    #動画ファイルから音声ファイルを作成
    convert_mts_to_wav(movie_folder, wav_folder)
    #音声ファイルからメルスペクトログラムを作成
    convert_audio_to_melspectrogram(wav_folder, mels_folder)
    
    # フォルダの中身を消去
    clear_folder(movie_folder)
    clear_folder(wav_folder)

if __name__ == "__main__":
    main()
    main()