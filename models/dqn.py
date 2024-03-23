import torch.nn as nn

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        # 3x3の盤面を入力とする1チャンネルの畳み込み層
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=2, stride=1),  # 出力サイズ: (32, 2, 2)
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 2 * 2, 64),  # 畳み込み層の出力を平坦化して全結合層に入力
            nn.ReLU(),
            nn.Linear(64, 9)  # 9つの行動（各セルに置く）
        )

    def forward(self, x):
        # 入力xを3x3の盤面の形式に変換
        x = x.view(-1, 1, 3, 3)  # (バッチサイズ, チャンネル数, 高さ, 幅)
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # 畳み込み層の出力を平坦化
        x = self.fc(x)
        return x