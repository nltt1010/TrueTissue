import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=2, freeze_backbone=False):
        """
        Khởi tạo mô hình phân loại ResNet18
        """
        super().__init__()
        # Load pretrained ResNet18
        weights = ResNet18_Weights.DEFAULT
        self.backbone = resnet18(weights=weights)
        
        # Đóng băng backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # Thay thế lớp FC
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)
        
    def get_cam_target_layer(self):
        """
        Lấy tầng Conv cuối cho Grad-CAM
        """
        return self.backbone.layer4[-1]