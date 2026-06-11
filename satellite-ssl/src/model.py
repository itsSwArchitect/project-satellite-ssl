
import torch
import torch.nn as nn
import torchvision.models as models

class SimCLRBackbone(nn.Module):
    def __init__(self, backbone_name="resnet18"):
        super().__init__()
        
        if backbone_name == "resnet18":
            backbone = models.resnet18(pretrained=False)
        else:
            backbone = models.resnet50(pretrained=False)
        
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim = backbone.fc.in_features
    
    def forward(self, x):
        features = self.backbone(x)
        features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)
        return features

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)

class SimCLRModel(nn.Module):
    def __init__(self, backbone_name="resnet18", projection_dim=128, hidden_dim=2048):
        super().__init__()
        
        self.backbone = SimCLRBackbone(backbone_name)
        self.projection = ProjectionHead(self.backbone.feature_dim, hidden_dim, projection_dim)
    
    def forward(self, x):
        features = self.backbone(x)
        projections = self.projection(features)
        return projections
    
    def get_features(self, x):
        return self.backbone(x)

def build_model(backbone="resnet18", projection_dim=128, hidden_dim=2048):
    return SimCLRModel(backbone, projection_dim, hidden_dim)
