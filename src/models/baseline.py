import torch.nn as nn
from torchvision.models import resnet50

class ContextEncoder(nn.Module):
    
    def __init__(self):
        super(ContextEncoder, self).__init__()
        self.model = resnet50(pretrained=True)
        self.model.fc = nn.Identity() # don't pool the data

    def forward(self, x):
        embs = self.model(x)
        return embs

class Predictor(nn.Module):
    
    def __init__(self):
        super(Predictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(),
            nn.ReLU(), 
            nn.BatchNorm1d(),
        )
        self.model = nn.Identity()
    

    def forward(self, x):
        output = self.model(x)
        return output

class TargetEncoder(nn.Module):
    
    def __init__(self):
        super(TargetEncoder, self).__init__()
        self.model = resnet50(pretrained=True)
        self.model.fc = nn.Identity() # don't pool the data
        
    
    def forward(self, x):
        output = self.model(x)

        return output