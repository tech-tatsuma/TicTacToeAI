import torch.nn as nn

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(9, 516),  # 3x3盤面を1次元に平坦化
            nn.ReLU(),
            nn.Linear(516, 128),
            nn.ReLU(),
            nn.Linear(128, 9)   # 9つの行動（各セルに置く）
        )
    
    def forward(self, x):
        return self.fc(x)