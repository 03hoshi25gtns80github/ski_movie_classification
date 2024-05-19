import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from data import get_data_loader
from model import get_model
from loss import NTXentLoss

def train(root_dir, batch_size, epochs, out_dim, temperature, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")  # GPUを使用しているかどうかを表示
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
            
            loss = criterion(z_i, z_j)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss}")

        # 10エポックごとにエンコーダを保存
        if (epoch + 1) % 2 == 0:
            save_dir = os.path.join(os.path.dirname(__file__), '../SimCLR')
            os.makedirs(save_dir, exist_ok=True)
            encoder_path = os.path.join(save_dir, f'SimCLR_encoder_bs{batch_size}_od{out_dim}_temp{temperature}_lr{learning_rate}_epoch{epoch+1}.pth')
            torch.save(model.state_dict(), encoder_path)
            print(f"Encoder saved to {encoder_path}")

    # 学習過程の可視化
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), loss_history, marker='o')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # スクリプトと同じフォルダに保存
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.join(script_dir, f'training_loss_bs{batch_size}_od{out_dim}_temp{temperature}_lr{learning_rate}.png')
    plt.savefig(plot_path)
    plt.show()
