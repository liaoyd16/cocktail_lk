
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
        # 256x128x8 -> 166x83x12
        self.upward_net2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=8, out_channels=8,\
                      padding=(0,0), stride=(3,3), kernel_size=(8,5), dilation=(2,2)),
            nn.ReLU(),
            ResBlock(8, 12),
            *[ResBlock(12, 12) for _ in range(keepsize_num_en)]
        )
        # 166x83x12 -> 108x54x16
        self.upward_net3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=12, out_channels=12,\
                      padding=(0,0), stride=(3,3), kernel_size=(5,3), dilation=(2,2)),
            nn.ReLU(),
            ResBlock(12, 16),
            *[ResBlock(16, 16) for _ in range(keepsize_num_en)]
        )
        # 108x54x16 -> 70x35x24
        self.upward_net4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=16, out_channels=16,\
                      padding=(0,0), stride=(3,3), kernel_size=(4,3), dilation=(2,2)),
            nn.ReLU(),
            ResBlock(16, 24),
            *[ResBlock(24, 24) for _ in range(keepsize_num_en)]
        )
        # 70x35x24 -> 46x23x32
        self.upward_net5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=24, out_channels=24,\
                      padding=(0,1), stride=(3,3), kernel_size=(2,3), dilation=(2,2)),
            nn.ReLU(),
            ResBlock(24, 32),
            *[ResBlock(32, 32) for _ in range(keepsize_num_en)]
        )
        # 46x23x32 -> 30x15x48
        self.upward_net6 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=32, out_channels=32,\
                      padding=(0,1), stride=(3,3), kernel_size=(2,3), dilation=(2,2)),
            nn.ReLU(),
            ResBlock(32, 48),
            *[ResBlock(48, 48) for _ in range(keepsize_num_en)]
        )
        # 30x15x48 -> 20x10x64
        self.upward_net7 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=48, out_channels=48,\
                      padding=(0,1), stride=(3,3), kernel_size=(2,3), dilation=(2,2)),
            nn.ReLU(),
            ResBlock(48, 64),
            *[ResBlock(64, 64) for _ in range(keepsize_num_en)]
        )
        # 20x10x64 -> 13x6x128
        self.upward_net8 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=64, out_channels=64,\
                      padding=(0,1), stride=(3,3), kernel_size=(2,3), dilation=(2,2)),
            nn.ReLU(),
            ResBlock(64, 128),
            *[ResBlock(128, 128) for _ in range(keepsize_num_en)]
        )
        # 13x6x128 -> 8x4x128
        self.upward_net9 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=128, out_channels=128,\
                      padding=(0,1), stride=(3,3), kernel_size=(2,3), dilation=(2,2)),
            nn.ReLU(),
            ResBlock(128, 128),
            *[ResBlock(128, 128) for _ in range(keepsize_num_en)]
        )

        self.fc1 = nn.Linear(4096, 256)
        self.fc2 = nn.Linear(256, 4096)

        # 8x4x128 -> 13x6x128
        self.downward_net9 = nn.Sequential(
            *[ResBlock(128, 128) for _ in range(keepsize_num_de)],
            ResBlock(128, 128),
            nn.ConvTranspose2d(in_channels=128, out_channels=128,\
                      padding=(0,1), stride=(3,3), kernel_size=(2,3), dilation=(2,2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2,2), stride=(2,2), padding=(1,0)),
        )

        # 13x6x128 -> 13x6x128 -> 20x10x64
        if shortcut[7]: self.uconv8 = ResBlock(256, 128)
        self.downward_net8 = nn.Sequential(
            *[ResBlock(128, 128) for _ in range(keepsize_num_de)],
            ResBlock(128, 64),
            nn.ConvTranspose2d(in_channels=64, out_channels=64,\
                      padding=(0,1), stride=(3,3), kernel_size=(2,3), dilation=(2,2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2,2), stride=(2,2), padding=(1,1))
        )

        # 20x10x64 -> 20x10x64 -> 30x15x48
        if shortcut[6]: self.uconv7 = ResBlock(128, 64)
        self.downward_net7 = nn.Sequential(
            *[ResBlock(64, 64) for _ in range(keepsize_num_de)],
            ResBlock(64, 48),
            nn.ConvTranspose2d(in_channels=48, out_channels=48,\
                      padding=(0,1), stride=(3,3), kernel_size=(2,3), dilation=(2,2)),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2,2), stride=(2,2))
        )

        # 30x15x48 -> 30x15x48 -> 46x23x32
        if shortcut[5]: self.uconv6 = ResBlock(96, 48)
        self.downward_net6 = nn.Sequential(
            *[ResBlock(48, 48) for _ in range(keepsize_num_de)],
            ResBlock(48, 32),
            nn.ConvTranspose2d(in_channels=32, out_channels=32,\
                      padding=(0,1), stride=(3,3), kernel_size=(2,3), dilation=(2,2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2,2), stride=(2,2), padding=(1,1))
        )

        # 46x23x32 -> 70x35x24
        if shortcut[4]: self.uconv5 = ResBlock(64, 32)
        self.downward_net5 = nn.Sequential(
            *[ResBlock(32, 32) for _ in range(keepsize_num_de)],
            ResBlock(32, 24),
            nn.ConvTranspose2d(in_channels=24, out_channels=24,\
                      padding=(0,1), stride=(3,3), kernel_size=(2,3), dilation=(2,2)),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2,2), stride=(2,2), padding=(1,1)),
        )

        # 70x35x24 -> 108x54x16
        if shortcut[3]: self.uconv4 = ResBlock(48, 24)
        self.downward_net4 = nn.Sequential(
            *[ResBlock(24, 24) for _ in range(keepsize_num_de)],
            ResBlock(24, 16),
            nn.ConvTranspose2d(in_channels=16, out_channels=16,\
                      padding=(0,0), stride=(3,3), kernel_size=(4,3), dilation=(2,2)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2,2), stride=(2,2), padding=(1,1))
        )

        # 108x54x16 -> 166x83x12
        if shortcut[2]: self.uconv3 = ResBlock(32, 16)
        self.downward_net3 = nn.Sequential(
            *[ResBlock(16, 16) for _ in range(keepsize_num_de)],
            ResBlock(16, 12),
            nn.ConvTranspose2d(in_channels=12, out_channels=12,\
                      padding=(0,0), stride=(3,3), kernel_size=(5,3), dilation=(2,2)),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2,2), stride=(2,2), padding=(1,1))
        )

        # 166x83x12 -> 256x128x8
        if shortcut[1]: self.uconv2 = ResBlock(24, 12)
        self.downward_net2 = nn.Sequential(
            *[ResBlock(12, 12) for _ in range(keepsize_num_de)],
            ResBlock(12, 8),
            nn.ConvTranspose2d(in_channels=8, out_channels=8,\
                      padding=(0,0), stride=(3,3), kernel_size=(8,5), dilation=(2,2)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2,2), stride=(2,2), padding=(1,1)),
        )

        # 256x128x8 -> 256x128x1
        if shortcut[0]: self.uconv1 = ResBlock(16, 8)
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
        self.x1 = self.upward_net1(specgrams)
        self.x2 = self.upward_net2(self.x1) * attentions[0]
        self.x3 = self.upward_net3(self.x2) * attentions[1]
        self.x4 = self.upward_net4(self.x3) * attentions[2]
        self.x5 = self.upward_net5(self.x4) * attentions[3]
        self.x6 = self.upward_net6(self.x5) * attentions[4]
        self.x7 = self.upward_net7(self.x6) * attentions[5]
        self.x8 = self.upward_net8(self.x7) * attentions[6]
        self.x9 = self.upward_net9(self.x8) * attentions[7]

        tops = F.relu(self.fc1(self.x9.view(-1, 4096) )) * attentions[8]
        self.x10 = tops
        return tops

    def downward(self, tops):
        x9 = self.fc2(tops).view(-1, 128, 8, 4)
        # 8x4x128 -> 13x6x128
        x8 = self.downward_net9(x9)
        
        # 13x6x128 -> 20x10x64
        if self.shortcut[7]:
            x8 = F.relu(self.uconv8(torch.cat((x8, self.x8), dim=1)))
        x7 = self.downward_net8(x8)
        
        # 20x10x64 -> 30x15x48
        if self.shortcut[6]:
            x7 = F.relu(self.uconv7(torch.cat((x7, self.x7), dim=1)))
        x6 = self.downward_net7(x7)
        
        # 30x15x48 -> 46x23x32
        if self.shortcut[5]:
            x6 = F.relu(self.uconv6(torch.cat((x6, self.x6), dim=1)))
        x5 = self.downward_net6(x6)
        
        # 46x23x32 -> 70x35x24
        if self.shortcut[4]:
            x5 = F.relu(self.uconv5(torch.cat((x5, self.x5), dim=1)))
        x4 = self.downward_net5(x5)
        
        # 70x35x24 -> 108x54x16
        if self.shortcut[3]:
            x4 = F.relu(self.uconv4(torch.cat((x4, self.x4), dim=1)))
        x3 = self.downward_net4(x4)
        
        # 108x54x16 -> 166x83x12
        if self.shortcut[2]:
            x3 = F.relu(self.uconv3(torch.cat((x3, self.x3), dim=1)))
        x2 = self.downward_net3(x3)
        
        # 166x83x12 -> 256x128x8
        if self.shortcut[1]:
            x2 = F.relu(self.uconv2(torch.cat((x2, self.x2), dim=1)))
        x1 = self.downward_net2(x2)
        
        # 256x128x8 -> 256x128x1
        if self.shortcut[0]:
            x1 = F.relu(self.uconv1(torch.cat((x1, self.x1), dim=1)))
        x0 = self.downward_net1(x1)

        return x0
