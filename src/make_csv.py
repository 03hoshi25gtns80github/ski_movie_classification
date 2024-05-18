import os
import csv

def create_dataset_csv(mels_folder, output_csv):
    # CSVファイルのヘッダー
    header = ['path', 'label']
    
    # CSVファイルを作成
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        
        # melsフォルダ内のすべてのサブフォルダを探索
        for subdir in os.listdir(mels_folder):
            subdir_path = os.path.join(mels_folder, subdir)
            
            if os.path.isdir(subdir_path):
                # サブフォルダ内のすべてのファイルを取得
                for filename in os.listdir(subdir_path):
                    if filename.endswith(".png"):
                        file_path = os.path.join(subdir_path, filename)
                        relative_path = os.path.relpath(file_path, start=os.path.dirname(output_csv))
                        label = subdir
                        
                        # CSVファイルに書き込み
                        writer.writerow([relative_path, label])
                        print(f"Added {relative_path} with label {label} to CSV")

# 使用例
mels_folder = "../mels"
output_csv = "../dataset/data.csv"
create_dataset_csv(mels_folder, output_csv)