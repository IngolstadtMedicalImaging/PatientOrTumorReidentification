import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import os

def get_model(args, n_classes: int, pretrained=True):
    model = MIL(n_classes, arch=args.arch, pretrained_backbone=args.pretrained_backbone)
    return model

def load_pretrained_model(args, n_classes):
    # os.environ["CUDA_VISIBLE_DEVICES"] = "MIG-88623a1c-c660-5e58-8722-cd90a957deee"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    model = get_model(args, n_classes=n_classes, pretrained=args.pretrained_model)

    if torch.cuda.is_available():
        pretrained_dict = torch.load(args.weights_dir)
    else:
        pretrained_dict = torch.load(args.weights_dir, map_location=torch.device('cpu')) # CPU Machines only

    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)

    return model


def load_pretrained_backbone(feature_extractor, pretrained_backbone):
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    if torch.cuda.is_available():
        pretrained_dict = torch.load(pretrained_backbone)
    else:
        pretrained_dict = torch.load(pretrained_backbone, map_location=torch.device('cpu')) # CPU Machines only

    for key in list(pretrained_dict.keys()):
        pretrained_dict[key.replace('feature_extractor.', '')] = pretrained_dict.pop(key)
    
    feature_extractor.load_state_dict(pretrained_dict)
    print([print(param) for param in feature_extractor.fc.parameters()])

    return feature_extractor


class MIL(nn.Module):
    def __init__(self, n_classes, arch, pretrained_backbone):
        super(MIL, self).__init__()
        self.L = 2048
        self.D = 128
        self.K = 1
        self.n_classes = n_classes

        if arch == 'mil-resnet18':
            self.feature_extractor = models.resnet18(pretrained=True)
        if arch == 'mil-resnet50':
            self.feature_extractor = models.resnet50(pretrained=True)
            self.feature_extractor.fc = nn.Linear(self.feature_extractor.fc.in_features, self.n_classes)
            if pretrained_backbone != 'none':
                self.feature_extractor = load_pretrained_backbone(self.feature_extractor, pretrained_backbone)
                print("loaded pretrained backbone")
                
        self.classifier = self.feature_extractor.fc
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

    def forward(self, x):
        batch_Y = []

        # for every bag of patches, do loop
        for bag in torch.tensor_split(x, x.shape[0]):
            
            x = torch.squeeze(bag)

            H = self.feature_extractor(x)
            H = torch.squeeze(H)

            A = self.attention(H)  # NxK
            A = torch.transpose(A, 1, 0)  # KxN
            A = F.softmax(A, dim=1)  # softmax over N

            M = torch.mm(A, H)  # KxL

            Y = self.classifier(M)

            batch_Y.append(Y)

        return torch.squeeze(torch.stack(batch_Y))