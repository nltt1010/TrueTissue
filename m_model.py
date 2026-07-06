import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

# =====================================================================
# [NHÓM 1] CỔNG CHÚ Ý (ATTENTION GATE)
# Tác dụng: Giúp Decoder chỉ tập trung vào tế bào từ Encoder chuyển sang, lọc bỏ nhiễu nền trắng.
# =====================================================================
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: Tín hiệu từ dưới Decoder đi lên | x: Tín hiệu từ Encoder đi sang
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi  # Nhân mask chú ý vào đặc trưng

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        
        # Encoder (Giữ nguyên Conv2d để lấy đặc trưng)
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, 64, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True))
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True))
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True))
        self.enc4 = nn.Sequential(nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True))
        
        # =====================================================================
        # [NHÓM 1] THAY THẾ DECONV BẰNG PIXEL SHUFFLE
        # Tác dụng: Thay vì dùng ConvTranspose2d nặng nề, ta dùng Conv2d thông thường 
        # nhưng tăng số kênh lên gấp 4 lần, rồi dùng PixelShuffle(2) để "dàn" các kênh 
        # đó ra thành không gian ảnh rộng gấp đôi. Tốc độ nhanh hơn, ảnh nét hơn.
        # =====================================================================
        self.dec1 = nn.Sequential(
            nn.Conv2d(512, 256 * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.att1 = AttentionGate(F_g=256, F_l=256, F_int=128) # Chú ý cho tầng 1
        
        self.dec2 = nn.Sequential(
            nn.Conv2d(512, 128 * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.att2 = AttentionGate(F_g=128, F_l=128, F_int=64) # Chú ý cho tầng 2

        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 64 * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.att3 = AttentionGate(F_g=64, F_l=64, F_int=32) # Chú ý cho tầng 3
        
        self.final = nn.Sequential(
            nn.Conv2d(128, out_channels * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.Tanh()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        d1 = self.dec1(e4)
        x3 = self.att1(g=d1, x=e3) # Dùng cổng chú ý lọc e3 trước khi ghép
        d2 = self.dec2(torch.cat([d1, x3], dim=1))
        
        x2 = self.att2(g=d2, x=e2) # Dùng cổng chú ý lọc e2 trước khi ghép
        d3 = self.dec3(torch.cat([d2, x2], dim=1))
        
        x1 = self.att3(g=d3, x=e1) # Dùng cổng chú ý lọc e1 trước khi ghép
        out = self.final(torch.cat([d3, x1], dim=1))
        
        return out

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=6):
        super().__init__()
        # =====================================================================
        # [NHÓM 3] SPECTRAL NORMALIZATION
        # Tác dụng: Kìm hãm sức mạnh của Discriminator. Kỹ thuật này trói buộc
        # trọng số (Lipschitz constraint) giúp GAN không bị sụp đổ (Mode Collapse).
        # =====================================================================
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1)), 
            nn.LeakyReLU(0.2),
            
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)), 
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            
            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)), 
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            
            spectral_norm(nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1))
        )
        
    def forward(self, x, target):
        return self.model(torch.cat([x, target], dim=1))