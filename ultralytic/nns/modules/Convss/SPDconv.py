import torch
import torch.nn.functional as F
import torch.nn as nn
from ultralytics.nn.modules.conv import *
class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int,
                 group_num: int = 16,
                 eps: float = 1e-10
                 ):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1))
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps
 
    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias
class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num:int, 
                 group_num:int = 16, 
                 eps:float = 1e-10
                 ):
        super(GroupBatchnorm2d,self).__init__()
        assert c_num    >= group_num
        self.group_num  = group_num
        self.gamma      = nn.Parameter(torch.randn(c_num, 1, 1))
        self.beta       = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps        = eps

    def forward(self, x):
        N, C, H, W  = x.size()
        x           = x.view(   N, self.group_num, -1   )
        mean        = x.mean(   dim = 2, keepdim = True )
        std         = x.std (   dim = 2, keepdim = True )
        x           = (x - mean) / (std+self.eps)
        x           = x.view(N, C, H, W)
        return x * self.gamma + self.beta
class SRU(nn.Module):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5,
                 torch_gn: bool = True
                 ):
        super().__init__()
 
        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(
            c_num=oup_channels, group_num=group_num)  ####组归一化
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()
 
    def forward(self, x):
        gn_x = self.gn(x)  ###归一化
        w_gamma = self.gn.weight / sum(self.gn.weight) #######
        w_gamma = w_gamma.view(1, -1, 1, 1)
        reweigts = self.sigomid(gn_x * w_gamma)
        # Gate
        w1 = torch.where(reweigts > self.gate_treshold, torch.ones_like(reweigts), reweigts)  # 大于门限值的设为1，否则保留原值
        w2 = torch.where(reweigts > self.gate_treshold, torch.zeros_like(reweigts), reweigts)  # 大于门限值的设为0，否则保留原值
        x_1 = w1 * x
        x_2 = w2 * x
        y = self.reconstruct(x_1, x_2)
        return y
 
    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)
class ALSS(nn.Module):
    def __init__(self, C_in, C_out, num_blocks=1, alpha=0.2, beta=1, stride=1, use_identity=False, shortcut_mode=False):
        super(ALSS, self).__init__()
        
        # Calculate split sizes
        self.shortcut_channels = int(C_in * alpha)
        self.main_in_channels = C_in - self.shortcut_channels
        bottleneck_channels = int(self.main_in_channels * beta)
        main_out_channels = C_out - self.shortcut_channels
        
        self.num_blocks = num_blocks
        
        # Shortcut path
        if stride == 2:
            if shortcut_mode == 0:
                self.shortcut = Conv(self.shortcut_channels, self.shortcut_channels, 3, 2)
            elif shortcut_mode == 1:
                self.shortcut = nn.Sequential(
                    nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                    Conv(self.shortcut_channels, self.shortcut_channels, 3, 1)
                )
            else:
                self.shortcut = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.shortcut = nn.Identity() if use_identity else \
                Conv(self.shortcut_channels, self.shortcut_channels, 3, 1)  ###DW
 
        
        # Main path
        self.initial_conv = Conv(self.main_in_channels, bottleneck_channels, 3, 1)
        
        self.middle_convs = nn.ModuleList()
        if stride == 2:
            self.middle_convs.append(Conv(bottleneck_channels, bottleneck_channels, 3, 2, g=bottleneck_channels, act=False))
            for _ in range(1, num_blocks):
                self.middle_convs.append(Conv(bottleneck_channels, bottleneck_channels, 3, 1, g=bottleneck_channels, act=False))
        else:
            for _ in range(num_blocks):
                self.middle_convs.append(Conv(bottleneck_channels, bottleneck_channels, 3, 1, g=bottleneck_channels, act=False))
        
        self.final_conv = Conv(bottleneck_channels, main_out_channels, 3, 1)
 
    def forward(self, x):
        # Split input into shortcut and main branches
        proportional_sizes=[self.shortcut_channels,self.main_in_channels]
        x = list(x.split(proportional_sizes, dim=1))
        # Process shortcut path
        shortcut_x = self.shortcut(x[0]) ##
        # Process main path
        main_x = self.initial_conv(x[1])
        for conv in self.middle_convs:
            main_x = conv(main_x)
        main_x = self.final_conv(main_x)
        out_x = torch.cat((main_x, shortcut_x), dim=1)
        out_x = channel_shuffle(out_x, 2)
        return out_x
class SPD_ALSS(nn.Module):
    def __init__(self,dim,C_in, C_out, num_blocks=1, alpha=0.2, beta=1, stride=1, use_identity=False, shortcut_mode=False):
        super().__init__()
        self.d = dimension
        self.shortcut_channels = int(C_in * alpha)
        self.main_in_channels = C_in - self.shortcut_channels
        bottleneck_channels = int(self.main_in_channels * beta)
        main_out_channels = C_out - self.shortcut_channels
    
        self.num_blocks = num_blocks
        
        # Shortcut path
        if stride == 2:
            if shortcut_mode == 0:
                self.shortcut = Conv(self.shortcut_channels, self.shortcut_channels, 3, 2)
            elif shortcut_mode == 1:
                self.shortcut = nn.Sequential(
                    nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                    Conv(self.shortcut_channels, self.shortcut_channels, 3, 1)
                )
            else:
                self.shortcut = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.shortcut = nn.Identity() if use_identity else \
                Conv(self.shortcut_channels, self.shortcut_channels, 3, 1)  ###DW
 
        
        # Main path
        self.initial_conv = Conv(self.main_in_channels, bottleneck_channels, 3, 1)
        
        self.middle_convs = nn.ModuleList()
        if stride == 2:
            self.middle_convs.append(Conv(bottleneck_channels, bottleneck_channels, 3, 2, g=bottleneck_channels, act=False))
            for _ in range(1, num_blocks):
                self.middle_convs.append(Conv(bottleneck_channels, bottleneck_channels, 3, 1, g=bottleneck_channels, act=False))
        else:
            for _ in range(num_blocks):
                self.middle_convs.append(Conv(bottleneck_channels, bottleneck_channels, 3, 1, g=bottleneck_channels, act=False))
        
        self.final_conv = Conv(bottleneck_channels, main_out_channels, 3, 1)
    def forward(self,x):
        x1=x[..., ::2, ::2]
        x2=x[..., 1::2, ::2]
        x3=x[..., ::2, 1::2]
        x4=x[..., 1::2, 1::2]
        main_x = torch.cat([x1, x2], dim=1)
        shortcut_x=torch.cat([x3, x4], dim=1)
        # Process shortcut path
        shortcut_x1 = self.shortcut(shortcut_x) ##
        main_x = self.initial_conv(main_x)
        
        
        
 

 
class space_to_depth(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, inc,ouc,dimension=1):
        super().__init__()
        self.d = dimension
        #self.conv = Conv(4*inc,ouc,k=1,s=1)
        self.conv=nn.Conv2d(inc*4,inc*4,kernel_size=1,stride=1,padding=0,bias=False)
       
    def forward(self, x):
         s=torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
         x= self.conv(s)
         return x
class Spd_Scconv(nn.Module):
    def __init__(self,
                 inc,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5,
                 torch_gn: bool = True
                 ):   ##dim通道数
        super(Spd_Scconv,self).__init__()
        self.gn = nn.GroupNorm(num_channels=inc, num_groups=group_num) if torch_gn else GroupBatchnorm2d(
            c_num=inc, group_num=group_num)  ####组归一化
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()
        
        self.spd=space_to_depth(inc,inc)
    def forward(self, x):
        gn_x = self.gn(x)  ###归一化
        w_gamma = self.gn.weight / sum(self.gn.weight) #######归一化权重
        w_gamma = w_gamma.view(1, -1, 1, 1)
        reweigts = self.sigomid(gn_x * w_gamma)
        # Gate
        w1 = torch.where(reweigts > self.gate_treshold, torch.ones_like(reweigts), reweigts)  # 大于门限值的设为1，否则保留原值
        w2 = torch.where(reweigts > self.gate_treshold, torch.zeros_like(reweigts), reweigts)  # 大于门限值的设为0，否则保留原值
        x_1 = w1 * x
        x_2 = w2 * x
        xx_1=self.spd(x_1)
        xx_2=self.spd(x_2)
        y = self.restruct(xx_1, xx_2)
        return y
    def restruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        final = torch.cat([x_11 + x_22, x_12 + x_21], dim=1)
        return final

class ScConv(nn.Module):
    def __init__(self,
                 op_channel: int,
                 group_num: int = 4,
                 gate_treshold: float = 0.5,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.SRU = SRU(op_channel,
                       group_num=group_num,
                       gate_treshold=gate_treshold)
        self.CRU = CRU(op_channel,
                       alpha=alpha,
                       squeeze_radio=squeeze_radio,
                       group_size=group_size,
                       group_kernel_size=group_kernel_size)
 
    def forward(self, x):
        x = self.SRU(x)
        x = self.CRU(x)
        return x
