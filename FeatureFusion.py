import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import math
import torch.nn.functional as F



class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=0, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        # self.bn = BatchNorm2d(out_chan)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU()
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)



class SegHead(nn.Module):
    '''
    两个卷积实现通道变换和特征图类别预测功能
    '''
    def __init__(self, in_chan, mid_chan, n_classes, *args, **kwargs):
        super(SegHead, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

class FFM(nn.Module):
    def __init__(self, in_cls, out_cls, use_att):
        super(FFM, self).__init__()

        # self.k = 64
        #
        self.use_att = use_att
        self.align = GUM(in_cls, out_cls)
        # self.query_linear = nn.Conv1d(in_channels=in_cls, out_channels=self.k, kernel_size=1)
        # self.key_linear = nn.Conv1d(in_channels=in_cls, out_channels=self.k, kernel_size=1)
        # self.value_linear = nn.Conv1d(in_channels=self.k, out_channels=out_cls, kernel_size=1)
        self.cca = CrossChannelAttention(in_channels=in_cls)
        self.csa = CrossSpatialAttention(kernel_size=7)
        self.fuse = nn.Sequential(
            nn.ReLU(),
            ConvBNReLU(out_cls, out_cls, padding=1),
            ConvBNReLU(out_cls, out_cls, padding=1)
        )
        self.bn = nn.BatchNorm2d(out_cls)
        self.relu = nn.ReLU()
        self.init()

    def forward(self, spatial, context):
        b, c, h, w = spatial.size()
        ctxt_upped = self.align(spatial, context)
        # avg_up1 = self.align(context, avg)
        # avg_up2 = self.align(spatial, avg)
        if self.use_att:
            cca_out = self.cca(spatial, ctxt_upped)
            csa_out = self.csa(cca_out, ctxt_upped)
            out = cca_out + csa_out
            # out = torch.cat((ctxt_upped, csa_out),dim=1)
        else:
            out = ctxt_upped + spatial #+ avg_up2
        out = self.fuse(out)
        if self.use_att:
            return out, csa_out
        else:
            return out, spatial

    def init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

class CrossChannelAttention(nn.Module):
    """
    CBAM混合注意力机制的通道注意力
    """

    def __init__(self, in_channels, ratio=16):
        super(CrossChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            # 利用1x1卷积代替全连接，避免输入必须尺度固定的问题，并减小计算量
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()
        self.init()

    def forward(self, sp, ctxt):
       avg_out = self.fc(self.avg_pool(ctxt))
       max_out = self.fc(self.max_pool(ctxt))
       out = avg_out + max_out
       out = self.sigmoid(out)
       return out * sp


    def init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()


class CrossSpatialAttention(nn.Module):
    """
    CBAM混合注意力机制的空间注意力
    """

    def __init__(self, kernel_size=3):
        super(CrossSpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.init()

    def forward(self, sp, ctxt):
        avg_out = torch.mean(ctxt, dim=1, keepdim=True)
        max_out, _ = torch.max(ctxt, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv1(out))
        return out * sp

    def init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()


class PAFM(nn.Module):
    def __init__(self, h_cls, l_cls):
        super(PAFM, self).__init__()
        # self.in_cls = in_cls
        self.conv_adp = ConvBNReLU(in_chan=l_cls, out_chan=h_cls, ks=1)
        self.conv = nn.Conv2d(2*h_cls, 2, kernel_size=1)
        self.init()

    def init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, high_scale, low_scale):
        low_scale = self.conv_adp(low_scale)
        cat = torch.cat((high_scale, low_scale), dim=1)
        pooled = nn.AdaptiveAvgPool2d((1, 1))(cat)
        pcat = pooled + cat
        pcat = self.conv(pcat)
        attn = nn.Softmax(dim=1)(pcat)

        attn0 = attn[:, 0, :, :].unsqueeze(1)
        attn1 = attn[:, 1, :, :].unsqueeze(1)
        oh = attn0 * high_scale
        ol = attn1 * low_scale
        out = oh + ol

        return out, attn0, attn1


class AttentionRefineModule(nn.Module):
    def __init__(self, in_cls, out_cls):
        super(AttentionRefineModule, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv_atten = nn.Conv2d(in_cls, in_cls, kernel_size=1, bias=False)
        self.use_adapt = (in_cls != out_cls)
        if self.use_adapt:
            self.conv_adpt = ConvBNReLU(in_cls, out_cls, ks=1)
        self.sigmoid = nn.Sigmoid()
        self.init_weight()

    def forward(self, x):
        o = self.avgpool(x)  # b,c,1,1
        atten = self.conv_atten(o)
        atten = self.sigmoid(atten)
        feat = torch.mul(x, atten)
        x = x + feat
        if self.use_adapt:
            x = self.conv_adpt(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class GUM(nn.Module):

    def __init__(self, inplane, outplane, kernel_size=3):
        super(GUM, self).__init__()

        self.conv1 = ConvBNReLU(2 * inplane, outplane, ks=3, padding=1)
        self.conv2 = ConvBNReLU(outplane, outplane, ks=3, padding=1)
        self.flow_make = nn.Conv2d(outplane, 2, kernel_size=kernel_size, padding=1, bias=False)

    def forward(self, spatial, context):

        context_origin = context
        sh, sw = spatial.size()[2:]
        size = (sh, sw)
        # ch, cw = context.size()[2:]
        # c_feature = self.down_c(context)
        # s_feature = self.down_s(spatial)

        context = F.upsample(context, size=size, mode="bilinear", align_corners=True)
        fuse = torch.cat((spatial, context), dim=1)
        fuse = self.conv1(fuse)
        fuse = self.conv2(fuse)
        # fuse = self.conv3(fuse)
        # fuse = self.conv4(fuse)
        # fuse =  c_feature + s_feature
        flow = self.flow_make(fuse)  # output is the prediction of the semantic flow field Dl-1
        upped_feat = self.flow_warp(context_origin, flow, size=size)

        return upped_feat

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()
        # n, c, h, w
        # n, 2, h, w

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output

class SPPM(nn.Module):
    def __init__(self, in_cls):
        super(SPPM, self).__init__()
        self.avg1 = nn.AdaptiveAvgPool2d(1)
        self.avg2 = nn.AdaptiveAvgPool2d(2)
        self.avg4 = nn.AdaptiveAvgPool2d(4)

        self.arm = AttentionRefineModule(in_cls, in_cls)
        out_cls = in_cls // 4

        self.conv1 = ConvBNReLU(in_cls, out_cls, ks=1)
        self.conv2 = ConvBNReLU(in_cls, out_cls, ks=1)
        self.conv4 = ConvBNReLU(in_cls, out_cls, ks=1)
        self.conv0 = ConvBNReLU(in_cls, out_cls, ks=1)

        self.conv_out = ConvBNReLU(in_cls, in_cls, ks=1)


    def forward(self, x):
        h, w = x.size()[2:]

        x = self.arm(x)

        arm_avg1 = self.conv1(self.avg1(x))
        arm_avg2 = self.conv2(self.avg2(x))
        arm_avg4 = self.conv4(self.avg4(x))
        arm_origin = self.conv0(x)

        arm_avg1 = F.interpolate(arm_avg1, (h,w), mode='bilinear', align_corners=True)
        arm_avg2 = F.interpolate(arm_avg2, (h,w), mode='bilinear', align_corners=True)
        arm_avg4 = F.interpolate(arm_avg4, (h,w), mode='bilinear', align_corners=True)


        arm_x = torch.cat((arm_origin, arm_avg4, arm_avg2, arm_avg1), dim=1)
        o = self.conv_out(arm_x)
        o = o + x
        return o

class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()      # b,c,h*w->b,h*w,c
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)  # b,c,h*w
        energy = torch.bmm(proj_query, proj_key)  # b,h*w,c . b,c,h*w -> b,h*w,h*w
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)  # b,c,h*w

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # b,c,h*w . b,h*w,h*w -> b,c,h*w
        out = out.view(m_batchsize, C, height, width)  # b,c,h*w->b,c,h,w

        out = self.gamma*out + x
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)  # b,c,h*w
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)  # b,c,h*w->b,h*w,c
        energy = torch.bmm(proj_query, proj_key)  # b,c,c
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy  # 为什么要执行这步操作
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out