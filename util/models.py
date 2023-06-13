#%%
import torch
from torch import nn

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x) -> None:
        batch, channel, _, _ = x.size()
        out = self.avg_pool(x).view(batch, channel)
        out = self.fc(out).view(batch, channel, 1, 1)
        out = x * out.expand_as(x)
        return out
    
class BottleNeckBlock(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1) -> None:
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        self.stage2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.stage3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels*self.expansion, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels*self.expansion),
            nn.ReLU(inplace=True),
        )
        
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * self.expansion, 1, stride, 0, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion),
        )

    def forward(self, x) -> None:
        out = self.stage1(x)
        out = self.stage2(out)
        out = self.stage3(out)
        residual = self.downsample(x)
        out = out + residual
        return out


#%%
class FeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stage1 = nn.Sequential(
            # [64, 3, 28, 28]
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.MaxPool2d(2)
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.MaxPool2d(2)
        )

        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        ) # [64, 128, 7, 7]

        
    def forward(self, x,) -> None:
        out = self.stage1(x)
        out = self.stage2(out)
        out = self.stage3(out)
        return out

#%%
class LabelClassifier(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        # [64, 1024, 7, 7]
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128*7*7, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 10),
        )
        
    def forward(self, x, ) -> None:
        out = x.view(x.size(0), -1)
        out = self.fc(out)
        return out


#%%
class NegativeGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        # ctx.save_for_backward(x) # 存取變數tensor
        ctx.lambda_ = lambda_ # lambda_是浮動的數值(np.float64)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # x = ctx.saved_tensors
        return grad_output.neg() * ctx.lambda_ , None


class DomainClassifier(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128*7*7, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 2),
        )
        
    def forward(self, x, lambda_) -> None:
        out = x.view(x.size(0), -1)
        out = NegativeGradient.apply(out, lambda_) 
        out = self.fc(out)
        return out


