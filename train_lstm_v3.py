# train_lstm_v3.py
import torch
import torch.nn as nn
import torch.optim as optim
from lstm_model import LSTMClassifier
from data_generator_v2 import generate_dataset_v2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# 生成数据
X, y = generate_dataset_v2(1000)  # 加大数据集
X = np.transpose(X, (0, 2, 1))  # (N, T, C)

# 标准化每条样本（每个通道减均值除方差）
X = (X - X.mean(axis=(1, 2), keepdims=True)) / (X.std(axis=(1, 2), keepdims=True) + 1e-6)

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为 TensorDataset 和 DataLoader
from torch.utils.data import TensorDataset, DataLoader

batch_size = 32
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                              torch.tensor(y_train, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                             torch.tensor(y_test, dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 初始化模型
model = LSTMClassifier(input_size=3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练过程
for epoch in range(20):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        outputs = model(xb)
        loss = criterion(outputs, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 5 == 0:
        print(f"第{epoch}轮，平均训练损失：{total_loss / len(train_loader):.4f}")

# 测试
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for xb, yb in test_loader:
        preds = model(xb)
        predicted = torch.argmax(preds, dim=1)
        all_preds.extend(predicted.numpy())
        all_labels.extend(yb.numpy())

acc = accuracy_score(all_labels, all_preds)
print(f"\n测试集准确率：{acc * 100:.2f}%")
