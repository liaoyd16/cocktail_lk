
import __init__
from __init__ import *
import torch.nn.init as init
from Blocks import PosNegLinear, ResBlock, ResTranspose

def initialize(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight)
        init.constant_(m.bias, 0)
    if isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight)

class ResAE(nn.Module):
    def __init__(self, keepsize_num_en, keepsize_num_de, shortcut):
        super(ResAE, self).__init__()

        self.shortcut = shortcut

        # 256x128x1 -> 256x128x8
        self.upward_net1 = nn.Sequential(
            ResBlock(1, 8),
            *[ResBlock(8, 8) for _ in range(keepsize_num_en)]
        )
        # 256x128x8 -> 128x64x16
        self.upward_net2 = nn.Sequential(
            # nn.MaxPool2d(kernel_size=(2,2), stride=2),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(2,2), stride=2),
            nn.ReLU(),
            ResBlock(8, 16),
            *[ResBlock(16, 16) for _ in range(keepsize_num_en)]
        )
        # 128x64x16 -> 64x32x32
        self.upward_net3 = nn.Sequential(
            # nn.MaxPool2d(kernel_size=(2,2), stride=2),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2,2), stride=2),
            nn.ReLU(),
            ResBlock(16, 32),
            *[ResBlock(32, 32) for _ in range(keepsize_num_en)]
        )
        # 64x32x32 -> 32x16x64
        self.upward_net4 = nn.Sequential(
            # nn.MaxPool2d(kernel_size=(2,2), stride=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(2,2), stride=2),
            nn.ReLU(),
            ResBlock(32, 64),
            *[ResBlock(64, 64) for _ in range(keepsize_num_en)]
        )
        # 32x16x64 -> 16x8x128
        self.upward_net5 = nn.Sequential(
            # nn.MaxPool2d(kernel_size=(2,2), stride=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(2,2), stride=2),
            nn.ReLU(),
            ResBlock(64, 128),
            *[ResBlock(128, 128) for _ in range(keepsize_num_en)]
        )
        # 16x8x128 -> 8x4x128
        self.upward_net6 = nn.Sequential(
            # nn.MaxPool2d(kernel_size=(2,2), stride=2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(2,2), stride=2),
            nn.ReLU(),
            ResBlock(128, 128),
            *[ResBlock(128, 128) for _ in range(keepsize_num_en)]
        )

        self.fc1 = nn.Linear(4096, 256)
        self.fc2 = nn.Linear(256, 4096)

        self.downward_net6 = nn.Sequential(
            *[ResBlock(128, 128) for _ in range(keepsize_num_de)],
            ResBlock(128, 128),
            ResTranspose(128, 128),
        )
        if shortcut[-1]: self.uconv5 = nn.Conv2d(256, 128, 3, 1, padding=1)
        self.downward_net5 = nn.Sequential(
            *[ResBlock(128, 128) for _ in range(keepsize_num_de)],
            ResBlock(128, 64),
            ResTranspose(64, 64),
        )
        if shortcut[-2]: self.uconv4 = nn.Conv2d(128, 64, 3, 1, padding=1)
        self.downward_net4 = nn.Sequential(
            *[ResBlock(64, 64) for _ in range(keepsize_num_de)],
            ResBlock(64, 32),
            ResTranspose(32, 32),
        )
        if shortcut[-3]: self.uconv3 = nn.Conv2d(64, 32, 3, 1, padding=1)
        self.downward_net3 = nn.Sequential(
            *[ResBlock(32, 32) for _ in range(keepsize_num_de)],
            ResBlock(32, 16),
            ResTranspose(16, 16),
        )
        if shortcut[-4]: self.uconv2 = nn.Conv2d(32, 16, 3, 1, padding=1)
        self.downward_net2 = nn.Sequential(
            *[ResBlock(16, 16) for _ in range(keepsize_num_de)],
            ResBlock(16, 8),
            ResTranspose(8, 8),
        )

        if shortcut[-5]: self.uconv1 = nn.Conv2d(16, 8, 3, 1, padding=1)
        self.downward_net1 = nn.Sequential(
            *[ResBlock(8, 8) for _ in range(keepsize_num_de)],
            ResBlock(8, 1),
            ResBlock(1, 1),
        )

        self.apply(initialize)

    def forward(self, specgrams, attentions):
        tops = self.upward(specgrams, attentions)
        recover = self.downward(tops)
        return recover

    def upward(self, specgrams, attentions):                # 1, 256, 128  # F.relu() bracket movement 19.12.5
        specgrams = specgrams.view(-1, 1, 256, 128)
        self.x1 = self.upward_net1(specgrams)                    # 8, 256, 128
        self.x2 = F.relu(self.upward_net2(self.x1)) * attentions[0]   # 16,128, 64
        self.x3 = F.relu(self.upward_net3(self.x2)) * attentions[1]   # 32, 64, 32
        self.x4 = F.relu(self.upward_net4(self.x3)) * attentions[2] # 64, 32, 16
        self.x5 = F.relu(self.upward_net5(self.x4)) * attentions[3] # 128,16, 8

        x6 = F.relu(self.upward_net6(self.x5)) * attentions[4] # 128, 8, 4

        # if not hasattr(self, 'pos_neg'): self.pos_neg = False
        # if not hasattr(self, 'relu_at_top'): self.relu_at_top = False

        self.x6 = x6.view(-1, 4096)                         # 2048+2048

        tops = F.relu(self.fc1(self.x6)) * attentions[5]
        self.x7 = tops
        return tops

    def downward(self, tops):
        x6 = self.fc2(tops).view(-1, 128, 8, 4)

        x5 = self.downward_net6(x6)
        if self.shortcut[-1]:
            x5 = F.relu(self.uconv5(torch.cat((x5, self.x5), dim=1)))

        x4 = self.downward_net5(x5)
        if self.shortcut[-2]:
            x4 = F.relu(self.uconv4(torch.cat((x4, self.x4), dim=1)))

        x3 = self.downward_net4(x4)
        if self.shortcut[-3]:
            x3 = F.relu(self.uconv3(torch.cat((x3, self.x3), dim=1)))

        x2 = self.downward_net3(x3)
        if self.shortcut[-4]:
            x2 = F.relu(self.uconv2(torch.cat((x2, self.x2), dim=1)))

        x1 = self.downward_net2(x2)
        if self.shortcut[-5]:
            x1 = F.relu(self.uconv1(torch.cat((x1, self.x1), dim=1)))

        x0 = self.downward_net1(x1)

        return x0
