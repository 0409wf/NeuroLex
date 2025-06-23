# lstm_model.py
import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=1, num_classes=2):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # 输入形状: (batch, seq_len, channels) -> 需要转成 (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 取最后一个时间点的输出
        out = self.fc(out)
        return out
