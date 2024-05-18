# スキーの映像のrenameをAIの予測で行う  

## c_src:教師無し学習の.py  
### main  
input:melsフォルダ内各フォルダ内のmels画像 
output:  
### data 
melsの読み込みとデータ拡張  
### loss  
損失関数定義  
### model  
エンコーダモデルと投影ヘッド定義  
### train  
学習過程定義  

## src:教師あり学習の.py  
### get_mels
input:movieフォルダ内のすべてのファイル  
output:入力動画のメルスペクトラム画像, melsフォルダ  
### make_csv  
input:melsフォルダ内各フォルダ内のmels画像  
output:csvメタデータ, datasetフォルダ  
### augment  
input:melsフォルダ内各フォルダ内のmels画像  
output:拡張したmels画像, augmentフォルダ  
### test_reset  
input:testフォルダ内の動画  
process:testフォルダ内のファイル名リセット 
### train  
input:dataset/data.csv  
output:model, modelフォルダ  
### inference  
input:testフォルダ内の動画  
process:ファイルrename  
### loder  
データローダー定義  
### model  
モデル定義  
create_modelのreturnをコメントアウトしてモデル選択  
### tekitou  
お試し用  

## フォルダ構成  
siwake/
├── augment/ # 水増しメルスペクトラム
├── dataset/ # データセットcsv
├── first_model/ # BERTモデルの完成版
├── mels/ # メルスペクトログラム画像
├── misc/ # 使わないmels
├── model/ # トレーニング済みモデル
├── movie/ # 学習用動画
├── src/ # ソースコード
├── test/ # テスト動画
├── wav/ # 音声ファイル
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── README.md
└── requirements.txt

