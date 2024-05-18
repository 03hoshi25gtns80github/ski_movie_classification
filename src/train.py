import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from loder import MelSpectrogramDataset
from model import create_model
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import os

# データローダの準備
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = MelSpectrogramDataset(csv_file='../dataset/data.csv', root_dir='../dataset', transform=transform)

# データセットの分割
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# モデルの定義
num_classes = len(set(dataset.annotations['label']))
model = create_model(num_classes=num_classes)

# 損失関数と最適化手法の定義
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 学習過程の記録用リスト
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

# 学習ループ
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for images, labels in train_dataloader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
    
    train_losses.append(running_loss / len(train_dataloader))
    train_accuracies.append(100 * correct_train / total_train)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_dataloader):.4f}, Train Accuracy: {100 * correct_train / total_train:.2f}%')

    # テストループ
    model.eval()
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for images, labels in test_dataloader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
    
    test_losses.append(test_loss / len(test_dataloader))
    test_accuracies.append(100 * correct_test / total_test)
    print(f'Test Loss: {test_loss / len(test_dataloader):.4f}, Test Accuracy: {100 * correct_test / total_test:.2f}%')

# 学習過程の可視化
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over Epochs')

plt.subplot(1, 3, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Train Accuracy over Epochs')

plt.subplot(1, 3, 3)
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Test Accuracy over Epochs')

plt.tight_layout()
plt.savefig('training_results.png')  # プロットをファイルに保存
plt.show()

# モデルの保存
model_dir = "../model"
model_path = os.path.join(model_dir, "trained_model.pth")
torch.save(model.state_dict(), model_path)

print("Training complete.")