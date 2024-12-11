import torch
from torch import nn
import torch.nn.functional as F

in_channels = 2

class EncoderLayer(nn.Module):
    def __init__(self, in_channels, hidden_size, kernel_size, stride, padding, dilation, use_layernorm, print_shape):
        super(EncoderLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels, hidden_size, kernel_size, stride, padding=padding, dilation=1)
        self.ln = nn.LayerNorm(hidden_size) if use_layernorm else None
        self.print_shape = print_shape
        
    def forward(self, x):
        x = self.conv(x.transpose(-1,-2))
        if self.print_shape:
            print('After Conv', x.shape)
        if self.ln is not None:
            x = self.ln(x.transpose(-1, -2))
        else:
            x = x.transpose(-1,-2)
        if self.print_shape:
            print('After Layernorm', x.shape)
        x = nn.functional.relu(x)
        return x
    
class ResidualBiGRU(nn.Module):
    def __init__(self, hidden_size, n_layers=1, bidir=True):
        super(ResidualBiGRU, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            n_layers,
            batch_first=True,
            bidirectional=bidir,
        )
        dir_factor = 2 if bidir else 1
        self.fc1 = nn.Linear(
            hidden_size * dir_factor, hidden_size * dir_factor * 2
        )
        self.ln1 = nn.LayerNorm(hidden_size * dir_factor * 2)
        self.fc2 = nn.Linear(hidden_size * dir_factor * 2, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, x, h=None):
        res, new_h = self.gru(x, h)
        # res.shape = (batch_size, sequence_size, 2*hidden_size)

        res = self.fc1(res)
        res = self.ln1(res)
        res = nn.functional.relu(res)

        res = self.fc2(res)
        res = self.ln2(res)
        res = nn.functional.relu(res)

        # skip connection
        res = res + x

        return res, new_h

class MultiResidualBiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, n_layers, bidir=True):
        super(MultiResidualBiGRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.n_layers = n_layers

        self.fc_in = nn.Linear(input_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)
        self.res_bigrus = nn.ModuleList(
            [
                ResidualBiGRU(hidden_size, n_layers=1, bidir=bidir)
                for _ in range(n_layers)
            ]
        )
        self.fc_out = nn.Linear(hidden_size, out_size)

    def forward(self, x, h=None):
        # if we are at the beginning of a sequence (no hidden state)
        if h is None:
            # (re)initialize the hidden state
            h = [None for _ in range(self.n_layers)]

        x = self.fc_in(x)
        x = self.ln(x)
        x = nn.functional.relu(x)

        new_h = []
        for i, res_bigru in enumerate(self.res_bigrus):
            x, new_hi = res_bigru(x, h[i])
            new_h.append(new_hi)

        x = self.fc_out(x)
#         x = F.normalize(x,dim=0)
        return x, new_h  # log probabilities + hidden states

class GRUNET(nn.Module):
    def __init__(self, arch, out_channels, kernel_size, stride, dconv_padding, hidden_size, n_layers, bidir=True, print_shape=False):
        super(GRUNET, self).__init__()

        self.input_size = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.hidden_size = hidden_size
        # self.out_size = out_size
        self.n_layers = n_layers
        self.padding = kernel_size//2
        self.print_shape = print_shape
        self.arch = arch
        self.dilation = 1
        assert arch[-1][1] == hidden_size

        self.conv = nn.Sequential(*[EncoderLayer(in_chan, out_chan, ksize, stride=stride, padding=ksize//2, dilation=self.dilation, use_layernorm=True, print_shape=print_shape) for in_chan, out_chan, stride, ksize in self.arch])
        self.res_bigrus = nn.ModuleList(
            [
                ResidualBiGRU(hidden_size, n_layers=1, bidir=bidir)
                for _ in range(n_layers)
            ]
        )
        self.dconv = nn.Sequential(*sum([[nn.ConvTranspose1d(out_chan, in_chan, ksize, stride=stride, padding=ksize//2, dilation=self.dilation, output_padding=1), 
                                  nn.Conv1d(in_chan, in_chan, ksize, stride=1, padding=ksize//2, dilation=self.dilation), nn.ReLU(), 
                                  nn.Conv1d(in_chan, in_chan, ksize, stride=1, padding=ksize//2, dilation=self.dilation), nn.ReLU()] for in_chan, out_chan, stride, ksize in reversed(arch)], []))
        self.output_layer = nn.Conv1d(2, 2, kernel_size=1, stride=1)
        
    def forward(self, x, h=None):
        # if we are at the beginning of a sequence (no hidden state)
        init_shape = x.shape
        if h is None:
            # (re)initialize the hidden state
            h = [None for _ in range(self.n_layers)]

        if self.print_shape:
            print('In', x.shape)
        x = self.conv(x)
        if self.print_shape:
            print('After EncoderLayer', x.shape)
        new_h = []
        for i, res_bigru in enumerate(self.res_bigrus):
            x, new_hi = res_bigru(x, h[i])
            new_h.append(new_hi)
        if self.print_shape:
            print('After GRU', x.shape)
        x = self.dconv(x.transpose(-1, -2))
        if self.print_shape:
            print('After DConv', x.shape)
            
        x = self.output_layer(x)
        x = x.transpose(-1,-2)
            
        if self.print_shape:
            print('After SmoothConv', x.shape)
        
        return x, new_h  # probabilities + hidden states