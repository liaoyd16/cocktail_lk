
import __init__
from __init__ import *

import Meta
import torch.nn as nn

class PosNegLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PosNegLinear, self).__init__()
        self.weights_pos2pos = nn.Parameter(torch.rand(out_dim, in_dim))
        self.weights_pos2neg = nn.Parameter(torch.rand(out_dim, in_dim))
        self.weights_neg2pos = nn.Parameter(torch.rand(out_dim, in_dim))
        self.weights_neg2neg = nn.Parameter(torch.rand(out_dim, in_dim))
        self.bias_pos = nn.Parameter(torch.rand(out_dim))
        self.bias_neg = nn.Parameter(torch.rand(out_dim))

        self.weights_pos2pos.requires_grad = True
        self.weights_pos2neg.requires_grad = True
        self.weights_neg2pos.requires_grad = True
        self.weights_neg2neg.requires_grad = True
        self.bias_pos.requires_grad = True
        self.bias_neg.requires_grad = True

    def forward(self, xs):
        x_pos = xs[0]
        x_neg = xs[1]
        y_pos = F.relu(F.linear(x_pos, (self.weights_pos2pos)**Meta.model_meta['POW'])\
                     - F.linear(x_neg, (self.weights_neg2pos)**Meta.model_meta['POW'])\
                     + self.bias_pos)
        y_neg = F.relu(F.linear(x_pos, (self.weights_pos2neg)**Meta.model_meta['POW'])\
                     - F.linear(x_neg, (self.weights_neg2neg)**Meta.model_meta['POW'])\
                     + self.bias_neg)
        return (y_pos, y_neg)

def odd(w):
    return list(np.arange(1, w, step=2, dtype='long'))

def even(w):
    return list(np.arange(0, w, step=2, dtype='long'))

# class ResBlock(nn.Module):
#     def __init__(self, channels_in, channels_out):
#         super(ResBlock, self).__init__()

#         self.channels_in = channels_in
#         self.channels_out = channels_out

#         self.conv1 = nn.Conv2d(in_channels=channels_in, out_channels=channels_out, kernel_size=(3,3), padding=1)
#         self.conv2 = nn.Conv2d(in_channels=channels_out, out_channels=channels_out, kernel_size=(3,3), padding=1)

#     def forward(self, x):
#         if self.channels_out > self.channels_in:
#             x1 = F.relu(self.conv1(x))
#             x1 =        self.conv2(x1)
#             x  = self.sizematch(self.channels_in, self.channels_out, x)
#             return F.relu(x + x1)
#         elif self.channels_out < self.channels_in:
#             x = F.relu(self.conv1(x))
#             x1 =       self.conv2(x)
#             return F.relu(x + x1)
#         else:
#             x1 = F.relu(self.conv1(x))
#             x1 =        self.conv2(x1)
#             return F.relu(x + x1)

#     def sizematch(self, channels_in, channels_out, x):
#         zeros = torch.zeros( (x.size()[0], channels_out - channels_in, x.shape[2], x.shape[3]), dtype=torch.float ).to(Meta.device)
#         return torch.cat((x, zeros), dim=1)

class ResBlock(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(ResBlock, self).__init__()

        self.channels_in = channels_in
        self.channels_out = channels_out

        self.residual_function = nn.Sequential(
            nn.BatchNorm2d(channels_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_out, channels_out, kernel_size=3, padding=1)
        )

        #shortcut
        self.shortcut = nn.Sequential()
        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if channels_in != channels_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(channels_in, channels_out, kernel_size=1, stride=1),
                nn.BatchNorm2d(channels_out),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResTranspose(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(ResTranspose, self).__init__()

        self.channels_in = channels_in
        self.channels_out = channels_out

        self.deconv1 = nn.ConvTranspose2d(in_channels=channels_in, out_channels=channels_out, kernel_size=(2,2), stride=2)
        self.deconv2 = nn.Conv2d(in_channels=channels_out, out_channels=channels_out, kernel_size=(3,3), padding=1)

    def forward(self, x):
        # cin = cout
        x1 = F.relu(self.deconv1(x))
        x1 =        self.deconv2(x1)
        x = self.sizematch(x)
        return F.relu(x + x1)

    def sizematch(self, x):
        # expand
        x2 = torch.zeros(x.shape[0], self.channels_in, x.shape[2]*2, x.shape[3]*2).to(Meta.device)

        row_x  = torch.zeros(x.shape[0], self.channels_in, x.shape[2], 2*x.shape[3]).to(Meta.device)
        row_x[:,:,:,odd(x.shape[3]*2)]   = x
        row_x[:,:,:,even(x.shape[3]*2)]  = x
        x2[:,:, odd(x.shape[2]*2),:] = row_x
        x2[:,:,even(x.shape[2]*2),:] = row_x

        return x2
