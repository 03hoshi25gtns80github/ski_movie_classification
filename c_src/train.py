import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from data import get_data_loader
from model import get_model
from loss import NTXentLoss

def train(root_dir, batch_size, epochs, out_dim, temperature, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = get_data_loader(root_dir, batch_size)  # drop_last=Trueを追加
    model = get_model(out_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = NTXentLoss(batch_size=batch_size, temperature=temperature)

    loss_history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for (x_i, x_j) in train_loader:
            if x_i.size(0) != batch_size or x_j.size(0) != batch_size:
                print(f"Skipping batch with size: x_i {x_i.size(0)}, x_j {x_j.size(0)}")
                continue
            
            x_i, x_j = x_i.to(device), x_j.to(device)
            z_i = model(x_i)
            z_j = model(x_j)
            
            # デバッグプリントを追加
            print(f"z_i shape: {z_i.shape}")
            print(f"z_j shape: {z_j.shape}")

            loss = criterion(z_i, z_j)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss}")

    # 学習過程の可視化
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), loss_history, marker='o')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # スクリプトと同じフォルダに保存
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.join(script_dir, 'training_loss.png')
    plt.savefig(plot_path)
    plt.show()
