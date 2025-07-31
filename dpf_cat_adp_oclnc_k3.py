import torch
import torch.nn as nn
from torch.nn import init
import math
import torch.nn.functional as F

from FeatureFusion import ConvBNReLU, FFM, SegHead, PAFM, AttentionRefineModule, SPPM, \
    CrossChannelAttention

from stdcnet import STDCNet813

'''
 pixel attention to aggrate contextual and spatial features and attention to fuse result
'''

class DPFNet(nn.Module):
    def __init__(self, backbone, num_classes=19):
        super(DPFNet, self).__init__()
        self.backbone = backbone

        sp2_inplanes = 32
        sp4_inplanes = 64
        sp8_inplanes = 256
        sp16_inplanes = 512
        sp32_inplanes = 1024
        arm_out_planes = 256
        out = 512

        self.ctxt_adp16 = AttentionRefineModule(sp16_inplanes, arm_out_planes)
        self.ctxt_adp4 = AttentionRefineModule(sp4_inplanes, arm_out_planes)
        self.ffm_high = FFM(arm_out_planes, arm_out_planes, False)
        self.head_high = SegHead(arm_out_planes, 128,  num_classes)

        # self.asppm = SPPM(sp32_inplanes)
        self.ctxt_adp32 = AttentionRefineModule(sp32_inplanes, arm_out_planes)
        self.ctxt_adp8 = AttentionRefineModule(sp8_inplanes, arm_out_planes)
        self.ffm_low = FFM(arm_out_planes, arm_out_planes, False)
        self.ffm_avg = FFM(arm_out_planes, arm_out_planes, False)
        self.ffm_32 = FFM(arm_out_planes, arm_out_planes, False)
        self.head_low = SegHead(arm_out_planes, 128, num_classes)

        self.cca_low = CrossChannelAttention(in_channels=arm_out_planes)
        self.cca_high = CrossChannelAttention(in_channels=arm_out_planes)

        # self.head4 = SegHead(sp4_inplanes, 64, num_classes)
        self.head8 = SegHead(arm_out_planes, 64, num_classes)
        self.head16 = SegHead(sp16_inplanes, 64, num_classes)
        self.head32 = SegHead(sp32_inplanes, 64, num_classes)

        self.conv_avg32 = AttentionRefineModule(sp32_inplanes, arm_out_planes)


        # self.pafm = PAFM(h_sp, l_sp)
        # self.head_fuse = SegHead(h_sp, h_sp, num_classes)
        self.gate = nn.Sequential(
            nn.Conv2d(2*num_classes, num_classes, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(num_classes),
            # nn.ReLU(inplace=False),
            # nn.Conv2d(num_classes, num_classes, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

        self.boundaryHead4 = SegHead(arm_out_planes, sp4_inplanes//2,1)
        self.boundaryHead8 = SegHead(arm_out_planes, sp8_inplanes//4, 1)
        self.boundaryHead4_o = SegHead(sp4_inplanes, sp4_inplanes // 2, 1)
        self.boundaryHead8_o = SegHead(sp8_inplanes, sp8_inplanes // 4, 1)
        self.boundaryHead2 = SegHead(sp2_inplanes, 32, 1)


        # out_cls = 1
        # self.f_spatial = nn.Sequential(
        #     nn.Conv2d(num_classes, out_cls, kernel_size=1, bias=False),
        #     nn.BatchNorm2d(out_cls)
        # )
        # self.f_context = nn.Sequential(
        #     nn.Conv2d(num_classes, out_cls, kernel_size=1, bias=False),
        #     nn.BatchNorm2d(out_cls)
        # )


    def forward(self, x):
        x2, x4, x8, x16_raw, x32_raw = self.backbone(x)
        b, c4, h4, w4 = x4.size()
        b, c8, h8, w8 = x8.size()
        _, _, h, w = x.size()


        avg32 = self.conv_avg32(x32_raw)
        avg32 = F.avg_pool2d(avg32, avg32.size()[2:])


        # x32 = self.asppm(x32)
        x32 = self.ctxt_adp32(x32_raw)
        x8 = self.ctxt_adp8(x8)
        # x32_up = F.interpolate(x32, (h8, w8), mode='bilinear', align_corners=True)
        x32_fuse, _ = self.ffm_avg(x32, avg32)
        fused_low, csa_out8 = self.ffm_low(x8, x32_fuse)
        fused_low = self.cca_low(fused_low, fused_low)
        seg_low = self.head_low(fused_low)
        seg_low = F.interpolate(seg_low, (h4, w4), mode='bilinear', align_corners=True)

        x16 = self.ctxt_adp16(x16_raw)
        x4 = self.ctxt_adp4(x4)

        # x16_fuse, _ = self.ffm_32(x16, x32_fuse)
        fused_high, csa_out4 = self.ffm_high(x4, x16)
        fused_high = self.cca_high(fused_high, fused_high)
        seg_high = self.head_high(fused_high)


        seg_cat = torch.cat((seg_high, seg_low), dim=1)
        g = self.gate(seg_cat)

        # p = nn.Softmax(dim=1)(seg_high)
        #

        # p = nn.Sigmoid()(seg_high)
        # g = 1-p
        seg_out = seg_low * g + seg_high * (1-g)

        seg_high = F.interpolate(seg_high, (h, w), mode='bilinear', align_corners=True)
        seg_low = F.interpolate(seg_low, (h, w), mode='bilinear', align_corners=True)

        seg_out = F.interpolate(seg_out, (h, w), mode='bilinear', align_corners=True)

        if self.training:
            e4 = self.boundaryHead4(x4)
            e8 = self.boundaryHead8(x8)
            # e4_o = self.boundaryHead4(csa_out4)
            # e8_o = self.boundaryHead8(csa_out8)
            e2 = self.boundaryHead2(x2)
            # seg4 = self.head4(x4)
            seg8 = self.head8(x8)
            seg16 = self.head16(x16_raw)
            seg32 = self.head32(x32_raw)
            # seg4 = F.interpolate(seg4, (h, w), mode='bilinear', align_corners=True)
            seg8 = F.interpolate(seg8, (h, w), mode='bilinear', align_corners=True)
            seg16 = F.interpolate(seg16, (h, w), mode='bilinear', align_corners=True)
            seg32 = F.interpolate(seg32, (h, w), mode='bilinear', align_corners=True)
            return seg_out, seg_high, seg_low, e4, e8, _, seg8, seg16, seg32, e2
        else:
            return seg_out, seg_high, seg_low, x4, csa_out4, x8, csa_out8, g, _, x16_raw





if __name__ == "__main__":
    # model = iteract1(num_classes=1000)
    backbone = STDCNet813()
    model = DPFNet(backbone, num_classes=19)
    model = model.cuda()
    model.eval()

    print(model)

    import torchsummary

    torchsummary.summary(model.cuda(), (3, 224, 224))

    from thop import profile, clever_format
    input = torch.zeros((1, 3, 224, 224)).cuda()
    flops, params = profile(model.cuda(), inputs=(input,))
    flops, params = clever_format([flops, params], '%.3f')

    print(f'参数量：{params}')
    print(f'FLOPs: {flops}')