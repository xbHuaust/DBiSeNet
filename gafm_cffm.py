import torch
import torch.nn as nn
from torch.nn import init
import math
import torch.nn.functional as F

from Hu_Segment.DPFNet.FeatureFusion import ConvBNReLU, FFM, SegHead, PAFM
# from Hu_Segment.DPFNet.FeatureAligns import GUM
from Hu_Segment.my_segment.models.interact_stdc import SPPM
from ThirdSegmentation.STDC_Seg.models.model_stages import FeatureFusionModule
from ThirdSegmentation.STDC_Seg.nets.stdcnet import STDCNet813

'''
 pixel attention to aggrate contextual and spatial features and attention to fuse result
'''

class DPFNet(nn.Module):
    def __init__(self, backbone, num_classes=19):
        super(DPFNet, self).__init__()
        self.backbone = backbone
        out = 256
        h_sp = 64
        h_txt = 512
        l_sp = 256
        l_txt = out

        self.ctxt_adp_low = ConvBNReLU(out, l_sp, ks=1)
        self.ctxt_adp_high = ConvBNReLU(h_txt, h_sp, ks=1)
        self.ffm_high = FFM(h_sp, h_sp)
        self.ffm_low = FFM(l_sp, l_sp)
        self.head_high = SegHead(h_sp, h_sp,  num_classes)
        self.head_low = SegHead(out, out // 4, num_classes)
        self.asppm = SPPM(1024, out)
        # self.pafm = PAFM(h_sp, l_sp)
        self.head_fuse = SegHead(h_sp, h_sp, num_classes)


    def forward(self, x):
        x2, x4, x8, x16, x32 = self.backbone(x)
        b, c4, h4, w4 = x4.size()
        b, c8, h8, w8 = x8.size()
        _, _, h, w = x.size()

        x16 = self.ctxt_adp_high(x16)
        fused_high = self.ffm_high(x4, x16)



        x32 = self.asppm(x32)
        fused_low = self.ffm_low(x8, x32)
        fused_low = F.interpolate(fused_low, (h4, w4), mode='bilinear', align_corners=True)

        # fused, att0, att1 = self.pafm(fused_high, fused_low)
        seg_out = self.head_fuse(fused)
        seg_out = F.interpolate(seg_out, (h, w), mode='bilinear', align_corners=True)



        if self.training:
            seg_high = self.head_high(fused_high)
            seg_high = F.interpolate(seg_high, (h, w), mode='bilinear', align_corners=True)
            seg_low = self.head_low(fused_low)
            seg_low = F.interpolate(seg_low, (h, w), mode='bilinear', align_corners=True)
            return seg_out, seg_high, seg_low, att0, att1
        else:
            return seg_out, att0, att1


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