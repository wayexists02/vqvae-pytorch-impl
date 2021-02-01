import torch
from torch import nn


class Quantize(nn.Module):

    def __init__(self, num_codewords, codebook_dim):
        super(Quantize, self).__init__()

        self.codebook = nn.parameter.Parameter(
            torch.randn(num_codewords, codebook_dim),
            requires_grad=True
        )

    def forward(self, x):
        N, C, H, W = x.size()
        _x = x.permute(0, 2, 3, 1).reshape(N*H*W, C)
        
        index = compute_index(_x, self.codebook)
        selected_codewords = torch.index_select(self.codebook, dim=0, index=index)
        selected_codewords = selected_codewords.view(N, H, W, C).permute(0, 3, 1, 2)

        return x + (selected_codewords - x).detach(), selected_codewords


class QuantizeEMA(nn.Module):

    def __init__(self, num_codewords, codebook_dim, gamma=0.99):
        super(QuantizeEMA, self).__init__()

        self.codebook = torch.randn(num_codewords, codebook_dim)/2
        self.gamma = gamma

        self.N = torch.ones(num_codewords)
        self.m = self.codebook.clone()

    def forward(self, x):
        with torch.no_grad():
            self.N = self.N.to(x.device)
            self.m = self.m.to(x.device)
            self.codebook = self.codebook.to(x.device)
            
            N, C, H, W = x.size()
            _x = x.permute(0, 2, 3, 1).reshape(N*H*W, C)

            index = compute_index(_x, self.codebook)
            selected_codewords = torch.index_select(self.codebook, dim=0, index=index)
            selected_codewords = selected_codewords.view(N, H, W, C).permute(0, 3, 1, 2)

            if self.training:
                new_N = torch.zeros_like(self.N).type(torch.FloatTensor).to(x.device)

                new_N.index_add_(dim=0, index=index, source=torch.ones_like(index).type(torch.FloatTensor).to(x.device))
                self.N[new_N > 0] = self.N[new_N > 0] * self.gamma + new_N[new_N > 0] * (1 - self.gamma)

                new_m = torch.zeros_like(self.m).type(torch.FloatTensor).to(x.device)

                new_m.index_add_(dim=0, index=index, source=_x)
                self.m[new_N > 0] = self.m[new_N > 0] * self.gamma + new_m[new_N > 0] * (1 - self.gamma)

                self.codebook = self.m / self.N.view(-1, 1)

        return x + (selected_codewords - x).detach()


class QuantizeChannelWise(nn.Module):

    def __init__(self, num_codewords, codebook_dim):
        super(QuantizeChannelWise, self).__init__()

        self.codebook = nn.parameter.Parameter(
            torch.randn(num_codewords, *codebook_dim),
            requires_grad=True
        )

    def forward(self, x):
        N, C, H, W = x.size()
        _x = x.view(N*C, 1, H, W)
        
        index = compute_index_channel_wise(_x, self.codebook)
        selected_codewords = torch.index_select(self.codebook, dim=0, index=index)
        selected_codewords = selected_codewords.view(N, C, H, W)

        return x + (selected_codewords - x).detach(), selected_codewords


class QuantizeEMAChannelWise(nn.Module):

    def __init__(self, num_codewords, codebook_dim, gamma=0.99):
        super(QuantizeEMAChannelWise, self).__init__()

        self.codebook = torch.randn(num_codewords, *codebook_dim)/2
        self.gamma = gamma

        self.N = torch.ones(num_codewords)
        self.m = self.codebook.clone()

    def forward(self, x):
        with torch.no_grad():
            self.N = self.N.to(x.device)
            self.m = self.m.to(x.device)
            self.codebook = self.codebook.to(x.device)
            
            N, C, H, W = x.size()
            _x = x.view(N*C, 1, H, W)

            index = compute_index_channel_wise(_x, self.codebook)
            selected_codewords = torch.index_select(self.codebook, dim=0, index=index)
            selected_codewords = selected_codewords.view(N, C, H, W)

            if self.training:
                new_N = torch.zeros_like(self.N).type(torch.FloatTensor).to(x.device)

                new_N.index_add_(dim=0, index=index, source=torch.ones_like(index).type(torch.FloatTensor).to(x.device))
                self.N[new_N > 0] = self.N[new_N > 0] * self.gamma + new_N[new_N > 0] * (1 - self.gamma)

                new_m = torch.zeros_like(self.m).type(torch.FloatTensor).to(x.device)

                new_m.index_add_(dim=0, index=index, source=_x)
                self.m[new_N > 0] = self.m[new_N > 0] * self.gamma + new_m[new_N > 0] * (1 - self.gamma)

                self.codebook = self.m / self.N.view(-1, 1, 1, 1)

        return x + (selected_codewords - x).detach()


def compute_index(x, codebook):
    x = x.unsqueeze(1)
    codebook = codebook.unsqueeze(0)

    dist2 = torch.sum((x - codebook)**2, dim=-1)
    index = dist2.argmin(1)
    return index


def compute_index_channel_wise(x, codebook):
    x = x.unsqueeze(1)
    codebook = codebook.unsqueeze(0)

    dist2 = torch.sum((x - codebook)**2, dim=(2, 3, 4))
    index = dist2.argmin(1)
    return index
