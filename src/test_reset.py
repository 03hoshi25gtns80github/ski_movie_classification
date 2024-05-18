import os

def rename_files_in_folder(folder_path):
    # フォルダ内のすべてのファイルを取得
    files = [f for f in os.listdir(folder_path) if f.endswith('.MTS')]
    files.sort()  # ファイル名をソートして順番を固定

    for i, filename in enumerate(files):
        old_path = os.path.join(folder_path, filename)
        new_filename = f"{i+1}.MTS"
        new_path = os.path.join(folder_path, new_filename)
        
        os.rename(old_path, new_path)
        print(f"Renamed {old_path} to {new_path}")

if __name__ == "__main__":
    folder_path = "../test"
    rename_files_in_folder(folder_path)