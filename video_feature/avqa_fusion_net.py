import torch.nn as nn
from .resnet_model import resnet18


class AVQA_Fusion_Net(nn.Module):
    def __init__(self):
        super(AVQA_Fusion_Net, self).__init__()

        # for features
        self.fc_a1 = nn.Linear(128, 512)
        self.fc_a2 = nn.Linear(512, 512)

        # visual
        self.visual_net = resnet18(pretrained=True)

        # combine
        self.fc1 = nn.Linear(1024, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 128)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(128, 2)
        self.relu4 = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_gl = nn.Linear(1024, 512)
        self.tanh = nn.Tanh()

    def forward(self, visual):

        ## visual, input: [16, 20, 3, 224, 224]
        (B, T, C, H, W) = visual.size()
        visual = visual.view(B * T, C, H, W)  # [320, 3, 224, 224]
        v_feat_out_res18 = self.visual_net(visual)  # [320, 512, 14, 14]

        return v_feat_out_res18
