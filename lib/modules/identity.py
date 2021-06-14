from torch import nn
from torch.nn import functional as F


class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=True
    ):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class IDModule(nn.Module):
    def __init__(
        self,
        in_channels=256,
        out_channels=128,
        num_ids=1000
    ):
        super(IDModule, self).__init__()

        self.in_channels = in_channels
        self.layers = nn.Sequential(
            BasicConv(in_channels, in_channels, 1, stride=1, padding=0, relu=True, bn=True),
            BasicConv(in_channels, in_channels, 1, stride=1, padding=0, relu=True, bn=True),
            BasicConv(in_channels, out_channels, 1, stride=1, padding=0, relu=False, bn=False)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(out_channels, num_ids)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        id_feature = self.layers(x)
        return id_feature

    def cross_entropy_loss(self, id_feature, labels):
        pooled_id_feature = self.avg_pool(id_feature).squeeze(-1).squeeze(-1)
        id_predictions = self.classifier(pooled_id_feature)
        loss = nn.CrossEntropyLoss(reduction="mean")(id_predictions, labels)
        return loss

    def triplet_loss(self, anchor_feature, pos_feature, neg_feature, margin=0.1):
        if neg_feature.shape[1] != anchor_feature.shape[1]:
            loss = 0
            for i in range(neg_feature.shape[1]):
                loss += nn.TripletMarginLoss(margin=margin, p=2, reduction="mean")(
                    self.avg_pool(anchor_feature),
                    self.avg_pool(pos_feature),
                    self.avg_pool(neg_feature[:, i, :, :, :]))
            loss = loss / neg_feature.shape[1]
        else:
            loss = nn.TripletMarginLoss(margin=margin, p=2, reduction="mean")(
                self.avg_pool(anchor_feature),
                self.avg_pool(pos_feature),
                self.avg_pool(neg_feature)
            )
        return loss
