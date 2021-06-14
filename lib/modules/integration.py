import torch
from torch import nn


class IntegrationModule(nn.Module):
    def __init__(
        self,
        min_iou=0.2,
        enhance_weight_max=1.0,
        reduce_weight_max=1.0
    ):
        super(IntegrationModule, self).__init__()
        self.min_iou = min_iou
        self.enhance_weight_max = enhance_weight_max
        self.reduce_weight_max = reduce_weight_max

    def forward(self, enhance_feature, reduce_feature, overlaps):
        enhance_weight = self.compute_weight(overlaps, self.enhance_weight_max, self.min_iou)
        reduce_weight = self.compute_weight(overlaps, self.reduce_weight_max, self.min_iou)
        return enhance_feature * enhance_weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) - \
            reduce_feature * reduce_weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    def compute_weight(self, ious, weight_max, iou_min):
        weight = weight_max * torch.min(torch.max((ious - iou_min) / (1.0 - iou_min), torch.zeros_like(ious)), torch.ones_like(ious))
        return weight
