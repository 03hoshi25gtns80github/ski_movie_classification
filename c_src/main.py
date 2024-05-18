from train import train

if __name__ == "__main__":
    batch_size = 32
    epochs = 2
    out_dim = 128
    temperature = 0.5
    learning_rate = 3e-4
    root_dir = '../mels'

    train(root_dir, batch_size, epochs, out_dim, temperature, learning_rate)