import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=2, freeze_backbone=False):
        """
        num_classes: Số lượng lớp phân loại (Mặc định = 2: Normal/Abnormal)
        freeze_backbone: Nếu True, sẽ đóng băng các tầng CNN đầu, chỉ train tầng fc cuối.
                         Giúp train siêu nhanh trên CPU.
        """
        super().__init__()
        # Load mô hình ResNet18 với trọng số pre-trained ImageNet chuẩn mới nhất
        weights = ResNet18_Weights.DEFAULT
        self.backbone = resnet18(weights=weights)
        
        # Nếu muốn đóng băng trọng số cũ để fine-tune siêu tốc
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # Thay thế tầng fc cuối cùng để phù hợp với bài toán 2 lớp
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),  # Chống overfitting
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)
        
    def get_cam_target_layer(self):
        """
        Hàm hỗ trợ lấy tầng Conv cuối cùng để phục vụ việc trích xuất Heatmap Grad-CAM 
        khoanh vùng mô bệnh trên giao diện Web sau này.
        """
        return self.backbone.layer4[-1]