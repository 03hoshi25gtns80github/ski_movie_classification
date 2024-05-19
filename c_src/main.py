from train import train

if __name__ == "__main__":
    epochs = 4
    root_dir = '../mels'

    # ハイパーパラメータの組み合わせ
    hyperparams = [
        {"batch_size": 32, "out_dim": 128, "temperature": 0.5, "learning_rate": 3e-4},
        {"batch_size": 64, "out_dim": 128, "temperature": 0.5, "learning_rate": 3e-4},
        # 他のハイパーパラメータの組み合わせを追加可能
    ]

    for params in hyperparams:
        train(root_dir, params["batch_size"], epochs, params["out_dim"], params["temperature"], params["learning_rate"])