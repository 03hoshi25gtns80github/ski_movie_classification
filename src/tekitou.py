from PIL import Image

def check_image_channels(image_path):
    # 画像を読み込む
    image = Image.open(image_path)
    # 画像のモードを取得
    mode = image.mode
    # チャンネル数を判定
    if mode == 'RGB':
        channels = 3
    elif mode == 'RGBA':
        channels = 4
    elif mode == 'L':
        channels = 1
    else:
        channels = 'Unknown'
    
    print(f"Image mode: {mode}")
    print(f"Number of channels: {channels}")

if __name__ == "__main__":
    image_path = "../mels/hoshi/hoshi (2)_melspectrogram.png"
    check_image_channels(image_path)

