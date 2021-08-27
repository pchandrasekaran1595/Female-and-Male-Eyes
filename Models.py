from torch import nn, optim
from torchvision import models

#####################################################################################################

class ResNet50(nn.Module):
    def __init__(self, train_full=False):
        super(ResNet50, self).__init__()

        self.model = models.resnet50(pretrained=True, progress=True)
        if not train_full:
            self.freeze()
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features=in_features, out_features=1)
    
    def freeze(self):
        for params in self.model.parameters():
            params.requires_grad = False
        
    def getOptimizer(self, lr=1e-3, wd=0):
        params = [p for p in self.parameters() if p.requires_grad]
        return optim.Adam(params, lr=lr, weight_decay=wd)
    
    def getPlateauScheduler(self, optimizer=None, patience=5, eps=1e-8):
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=patience, eps=eps)
    
    def forward(self, x):
        return self.model(x)
    
#####################################################################################################

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        self.model = models.resnet50(pretrained=True, progress=True)
        self.model = nn.Sequential(*[*self.model.children()][:-1])
        self.model.add_module("Flatten", nn.Flatten())
    
    def forward(self, x):
        return self.model(x)
    
#####################################################################################################

class FeatureClassifier(nn.Module):
    def __init__(self):
        raise NotImplementedError("Model not Implemented")
    
    def forward(self, x):
        return None
    
#####################################################################################################
