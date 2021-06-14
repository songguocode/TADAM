import torch
import torch.nn as nn


class Memory(object):
    def __init__(self, initial_feature, memory_net):
        self.h_state = initial_feature

    def update(self, new_feature, memory_net):
        self.h_state = memory_net(new_feature, self.h_state)

    def train_update(self, feature_sequence, memory_net):
        for i, f in enumerate(feature_sequence):
            if i == 0:
                h = f
            else:
                h = memory_net(f, h)
        self.h_state = h


class MemoryNet(nn.Module):
    def __init__(
        self,
        feature_size=(256, 7, 7),
        num_ids=1000,
        kernel_size=(1, 1),
        bias=True,
    ):
        super(MemoryNet, self).__init__()
        C, H, W = feature_size
        self.init_state = nn.Conv2d(
            in_channels=C,
            out_channels=C,
            kernel_size=kernel_size,
            padding=(kernel_size[0] // 2, kernel_size[1] // 2),
            bias=bias
        )
        self.loss_classifier = nn.Linear(feature_size[0], num_ids)
        self.cell = ConvGRUCell(feature_size=feature_size, kernel_size=kernel_size, bias=bias)

    def forward(self, new_feature, state):
        return self.cell(new_feature, state)

    # To train memory, compute after update
    def memory_loss(self, feature, gt_id, pos_feature=None, neg_feature=None):
        pooled_feature = nn.AdaptiveAvgPool2d(1)(feature).squeeze(-1).squeeze(-1)
        prediction = self.loss_classifier(pooled_feature)
        crossentropy_loss = nn.CrossEntropyLoss(reduction='mean')(prediction, gt_id)
        if pos_feature is not None and neg_feature is not None:
            pooled_pos = nn.AdaptiveAvgPool2d(1)(pos_feature).squeeze(-1).squeeze(-1)
            pooled_neg = nn.AdaptiveAvgPool2d(1)(neg_feature).squeeze(-1).squeeze(-1)
            triplet_loss = nn.TripletMarginLoss(reduction='mean')(pooled_feature, pooled_pos, pooled_neg)
            return crossentropy_loss + triplet_loss
        else:
            return crossentropy_loss


class ConvGRUCell(nn.Module):
    def __init__(
        self,
        feature_size=(256, 7, 7),
        kernel_size=(3, 3),
        bias=True,
    ):
        super(ConvGRUCell, self).__init__()
        self.C, self.H, self.W = feature_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.hidden_dim = self.C
        self.bias = bias

        self.conv_gates = nn.Conv2d(
            in_channels=self.C + self.hidden_dim,
            out_channels=self.hidden_dim * 2,  # For update_gate, reset_gate
            kernel_size=kernel_size,
            padding=self.padding,
            bias=self.bias
        )

        self.conv_candidate = nn.Conv2d(
            in_channels=self.C + self.hidden_dim,
            out_channels=self.hidden_dim,  # For candidate neural memory
            kernel_size=kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, new_feature, h_prev):
        combined = torch.cat([new_feature, h_prev], dim=1)
        gates = self.conv_gates(combined)
        reset_gate, update_gate = torch.split(gates, self.hidden_dim, dim=1)
        reset_gate = torch.sigmoid(reset_gate)
        update_gate = torch.sigmoid(update_gate)
        combined = torch.cat([new_feature, reset_gate * h_prev], dim=1)
        candidate = torch.tanh(self.conv_candidate(combined))
        h_new = (1 - update_gate) * h_prev + update_gate * candidate
        return h_new
