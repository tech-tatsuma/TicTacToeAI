import torch.nn as nn

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=2, stride=1),  
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 2 * 2, 64),  
            nn.ReLU(),
            nn.Linear(64, 9)  
        )

    def forward(self, x):
        x = x.view(-1, 1, 3, 3) 
        x = self.conv(x)
        x = x.view(x.size(0), -1) 
        x = self.fc(x)
        return x