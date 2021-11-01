
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

        # 256x128x1 -> 256x128x16
        self.upward_net1 = nn.Sequential(
            ResBlock(1, 16),
            *[ResBlock(16, 16) for _ in range(keepsize_num_en)]
        )
        # 256x128x16 -> 86x43x64
        self.upward_net2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16,\
                      padding=(1,1), stride=(3,3), kernel_size=(4,4), dilation=(1,1)),
            nn.ReLU(),
            ResBlock(16, 64),
            *[ResBlock(64, 64) for _ in range(keepsize_num_en)]
        )
        # 86x43x64 -> 28x14x128
        self.upward_net3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64,\
                      padding=(0,0), stride=(3,3), kernel_size=(4,4), dilation=(1,1)),
            nn.ReLU(),
            ResBlock(64, 128),
            *[ResBlock(128, 128) for _ in range(keepsize_num_en)]
        )
        # 28x14x128 -> 8x4x128
        self.upward_net4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128,\
                      padding=(0,0), stride=(3,3), kernel_size=(5,4), dilation=(1,1)),
            nn.ReLU(),
            ResBlock(128, 128),
            *[ResBlock(128, 128) for _ in range(keepsize_num_en)]
        )

        self.fc1 = nn.Linear(4096, 256)
        self.fc2 = nn.Linear(256, 4096)

        self.downward_net4 = nn.Sequential(
            *[ResBlock(128, 128) for _ in range(keepsize_num_de)],
            ResBlock(128, 128),
            nn.ConvTranspose2d(in_channels=128, out_channels=128,\
                      padding=(0,0), stride=(3,3), kernel_size=(7,5), dilation=(1,1)),
            nn.ReLU()
        )
        if shortcut[-1]: self.uconv3 = ResBlock(256, 128)
        self.downward_net3 = nn.Sequential(
            *[ResBlock(128, 128) for _ in range(keepsize_num_de)],
            ResBlock(128, 64),
            nn.ConvTranspose2d(in_channels=64, out_channels=64,\
                      padding=(0,0), stride=(3,3), kernel_size=(4,4), dilation=(1,1)),
            nn.ReLU()
        )
        if shortcut[-2]: self.uconv2 = ResBlock(128, 64)
        self.downward_net2 = nn.Sequential(
            *[ResBlock(64, 64) for _ in range(keepsize_num_de)],
            ResBlock(64, 16),
            nn.ConvTranspose2d(in_channels=16, out_channels=16,\
                      padding=(1,1), stride=(3,3), kernel_size=(6,4), dilation=(1,1)),
            nn.ReLU()
        )
        if shortcut[-3]: self.uconv1 = ResBlock(32, 16)
        self.downward_net1 = nn.Sequential(
            *[ResBlock(16, 16) for _ in range(keepsize_num_de)],
            ResBlock(16, 1),
            ResBlock(1, 1)
        )

        self.apply(initialize)

    def forward(self, specgrams, attentions):
        tops = self.upward(specgrams, attentions)
        recover = self.downward(tops)
        return recover

    def upward(self, specgrams, attentions):                # 1, 256, 128  # F.relu() bracket movement 19.12.5
        specgrams = specgrams.view(-1, 1, 256, 128)
        self.x1 = self.upward_net1(specgrams)               # 8, 256, 128
        self.x2 = self.upward_net2(self.x1) * attentions[0] # 32, 85, 43
        self.x3 = self.upward_net3(self.x2) * attentions[1] # 128,28, 14
        self.x4 = self.upward_net4(self.x3) * attentions[2] # 128, 8, 4

        tops = F.relu(self.fc1(self.x4.view(-1, 4096) )) * attentions[3]
        self.x5 = tops
        return tops

    def downward(self, tops):
        x4 = self.fc2(tops).view(-1, 128, 8, 4)

        x3 = self.downward_net4(x4)

        if self.shortcut[-1]:
            x3 = F.relu(self.uconv3(torch.cat((x3, self.x3), dim=1)))
        x2 = self.downward_net3(x3)

        if self.shortcut[-2]:
            x2 = F.relu(self.uconv2(torch.cat((x2, self.x2), dim=1)))
        x1 = self.downward_net2(x2)

        if self.shortcut[-3]:
            x1 = F.relu(self.uconv1(torch.cat((x1, self.x1), dim=1)))
        x0 = self.downward_net1(x1)

        # print(x0.shape, x1.shape, x2.shape, x3.shape, x4.shape)
        return x0
