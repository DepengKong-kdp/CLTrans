
import torch
import torch.nn as nn
from Utils.Backbone_model import resnet18

class CLTrans(nn.Module):
    def __init__(self, input_size=1024, dim=64, pred_dim=16):
        super(CLTrans, self).__init__()

        self.backbone = resnet18(out_dims=32)
        self.backbone = nn.Sequential(*[p for p in self.backbone.children()][:-1])

        # build a 3-layer projector
        ConvDim = self.get_linear_size(torch.zeros(1, 1, input_size))
        print('Size of the encoded features: {}'.format(ConvDim))
        self.projector = nn.Sequential(
            nn.Linear(ConvDim, ConvDim // 2, bias=False),
            nn.BatchNorm1d(ConvDim // 2),
            nn.ReLU(inplace=True),  # First layer

            nn.Linear(ConvDim // 2, ConvDim // 2, bias=False),
            nn.BatchNorm1d(ConvDim // 2),
            nn.ReLU(inplace=True),  # Second layer

            nn.Linear(ConvDim // 2, dim, bias=False),  # Output layer
            nn.BatchNorm1d(dim)
        )

        # build a 2-layer predictor
        self.predictor = nn.Sequential(
            nn.Linear(dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),

            nn.Linear(pred_dim, dim)
        )
        self.count_parameters()

    def forward(self, x1, x2):
        '''
                Input:
                    x1: first augmentation of raw data
                    x2: second augmentation of the raw data
                Output:
                    p1, p2, z1, z2 = predictors and targets of the network
                '''
        z1_ = self.backbone(x1.unsqueeze(1))
        z1 = self.projector(torch.flatten(z1_, start_dim=1))

        z2_ = self.backbone(x2.unsqueeze(1))
        z2 = self.projector(torch.flatten(z2_, start_dim=1))

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        return p1, p2, z1.detach(), z2.detach()

    def get_linear_size(self, x):
        out = self.backbone(x)
        return out.size(1) * out.size(2)

    def count_parameters(self):
        total_num = 0
        total_num += sum(p.numel() for p in self.backbone.parameters())
        total_num += sum(p.numel() for p in self.projector.parameters())
        total_num += sum(p.numel() for p in self.predictor.parameters())

        print('Total number of model parameters: ', total_num)

class transfer_model(nn.Module):
    def __init__(self, dataset_name_pretrain, load_params):
        super(transfer_model, self).__init__()
        class_num = {'CWRU': 10, 'MFPT':3}

        cltrans= CLTrans()

        if load_params:
            cltrans.load_state_dict(torch.load('./params/CLTrans_{}-best-BN.pkl'.format(dataset_name_pretrain)))
        else:
            cltrans.eval()

        # feature encoder
        self.backbone = cltrans.backbone

        self.fc    = nn.Sequential(
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, class_num[dataset_name_pretrain])
        )
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, inputs):
        feature = self.backbone(inputs.unsqueeze(1))
        output = feature.view(feature.size(0), -1)
        output = self.fc(output)

        return output, feature

if __name__ == '__main__':
    model = CLTrans()
    print([p for p in model.backbone.children()][0])