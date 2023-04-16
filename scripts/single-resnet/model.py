import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def get_model(args, n_classes: int, pretrained=True):
    if args.arch == 'resnet18-pretraining':
        model = renset18_pretraining(n_classes)
    elif args.arch == 'resnet50-pretraining':
        model = renset50_pretraining(n_classes)
    return model


def load_pretrained_model(args, n_classes):
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    model = get_model(args, n_classes=n_classes, pretrained=args.pretrained_model)

    if torch.cuda.is_available():
        pretrained_dict = torch.load(args.weights_dir)
    else:
        pretrained_dict = torch.load(args.weights_dir, map_location=torch.device('cpu')) # CPU Machines only

    model.load_state_dict(pretrained_dict)

    # # Train only last layer from init
    # for param in model.parameters():
    #     param.requires_grad = False

    # for param in model.feature_extractor.fc.parameters():
    #     param.requires_grad = True
    
    # model.feature_extractor.fc.reset_parameters()

    return model



class renset50_pretraining(nn.Module):
    def __init__(self, n_classes):
        super(renset50_pretraining, self).__init__()
        self.n_classes = n_classes

        self.feature_extractor = models.resnet50(pretrained=True)
        self.feature_extractor.fc = nn.Linear(self.feature_extractor.fc.in_features, self.n_classes)

    def forward(self, x):
        Y = self.feature_extractor(x)
        return Y
    

class renset18_pretraining(nn.Module):
    def __init__(self, n_classes):
        super(renset18_pretraining, self).__init__()
        self.n_classes = n_classes

        self.feature_extractor = models.resnet18(pretrained=True)
        self.feature_extractor.fc = nn.Linear(self.feature_extractor.fc.in_features, self.n_classes)

    def forward(self, x):
        Y = self.feature_extractor(x)
        return Y

