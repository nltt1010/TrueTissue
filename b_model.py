import torch
import torch.nn as nn

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        
        # Encoder (Giảm chiều dữ liệu: 256 -> 128 -> 64 -> 32 -> 16)
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Decoder (Tăng kích thước đối xứng: 16 -> 32 -> 64 -> 128 -> 256)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Đường Encoder đi xuống
        e1 = self.enc1(x)     # Kích thước đầu ra: [B, 64, 128, 128]
        e2 = self.enc2(e1)    # Kích thước đầu ra: [B, 128, 64, 64]
        e3 = self.enc3(e2)    # Kích thước đầu ra: [B, 256, 32, 32]
        e4 = self.enc4(e3)    # Kích thước đầu ra: [B, 512, 16, 16]
        
        # Đường Decoder đi lên kèm Skip Connections gộp các đặc trưng không gian
        d1 = self.dec1(e4)    # Tăng lên kích thước: [B, 256, 32, 32]
        d2 = self.dec2(torch.cat([d1, e3], dim=1))  # Gộp 256+256 kênh -> [B, 128, 64, 64]
        d3 = self.dec3(torch.cat([d2, e2], dim=1))  # Gộp 128+128 kênh -> [B, 64, 128, 128]
        
        out = self.final(torch.cat([d3, e1], dim=1)) # Gộp 64+64 kênh -> [B, 3, 256, 256]
        return out

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=6):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        )
    def forward(self, x, target):
        return self.model(torch.cat([x, target], dim=1))