import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import StainingDataset
from pathlib import Path
from m_model import UNetGenerator, PatchDiscriminator 

from torchmetrics.image import StructuralSimilarityIndexMeasure 


DATA_ROOT = Path(r"G:/CV/pj/dataset/train")
CHECKPOINT_DIR = Path(r"G:/CV/pj/m_checkpoints") 
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_FILE_PATH = "./m_log/train_log.txt"

train_loader = DataLoader(StainingDataset(DATA_ROOT), batch_size=4, shuffle=True)

gen = UNetGenerator().to(device)
disc = PatchDiscriminator().to(device)
opt_g = optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
opt_d = optim.Adam(disc.parameters(), lr=0.0002, betas=(0.5, 0.999))
criterion_gan = nn.BCEWithLogitsLoss()
criterion_l1 = nn.L1Loss()

# =====================================================================
# [NHÓM 6] SSIM LOSS (Cấu trúc tương đồng)
# Khởi tạo bộ đo SSIM dùng làm loss. Ta đưa ảnh về dải [0,1].
# =====================================================================
ssim_module = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

start_epoch = 0
RESUME_CHECKPOINT = CHECKPOINT_DIR / "gen_last.pth" 
if RESUME_CHECKPOINT.exists():
    checkpoint = torch.load(RESUME_CHECKPOINT, map_location=device)
    gen.load_state_dict(checkpoint['gen'])
    start_epoch = checkpoint['epoch'] + 1
    if 'disc' in checkpoint: disc.load_state_dict(checkpoint['disc'])
    if 'opt_g' in checkpoint: opt_g.load_state_dict(checkpoint['opt_g'])
    if 'opt_d' in checkpoint: opt_d.load_state_dict(checkpoint['opt_d'])
    print(f"\n[RESUME] Nạp thành công! Tiếp tục từ Epoch {start_epoch}\n")

for epoch in range(start_epoch, 10):
    gen.train()
    for i, (real_in, real_out) in enumerate(train_loader):
        real_in, real_out = real_in.to(device), real_out.to(device)
        
        # =====================================================================
        # TỐI ƯU DISCRIMINATOR
        # =====================================================================
        opt_d.zero_grad()
        fake_out = gen(real_in)
        d_real = disc(real_in, real_out)
        d_fake = disc(real_in, fake_out.detach())
        
        # [NHÓM 3] LABEL SMOOTHING (Làm mềm nhãn)
        # Thay vì ép Discriminator đoán chuẩn 1.0 (Thật) và 0.0 (Giả), ta cho một dải ngẫu nhiên.
        # Giúp Discriminator bớt khắc nghiệt, Generator dễ thở hơn.
        real_labels = torch.empty_like(d_real).uniform_(0.9, 1.0).to(device)
        fake_labels = torch.empty_like(d_fake).uniform_(0.0, 0.1).to(device)
        
        d_loss = criterion_gan(d_real, real_labels) + criterion_gan(d_fake, fake_labels)
        d_loss.backward()
        opt_d.step()
        
        # =====================================================================
        # TỐI ƯU GENERATOR
        # =====================================================================
        opt_g.zero_grad()
        d_fake_for_g = disc(real_in, fake_out)
        
        # GAN Loss (Đánh lừa discriminator)
        loss_G_bce = criterion_gan(d_fake_for_g, torch.ones_like(d_fake_for_g)) 
        # L1 Loss (Khớp Pixel)
        loss_G_l1 = criterion_l1(fake_out, real_out) 
        
        # [NHÓM 6] KÍCH HOẠT SSIM LOSS
        # Chuyển dải [-1, 1] về [0, 1] để đo chuẩn
        fake_01 = (fake_out + 1) / 2.0
        real_01 = (real_out + 1) / 2.0
        ssim_val = ssim_module(fake_01, real_01)
        # Vì SSIM càng lớn (gần 1) càng tốt, nên loss sẽ là (1 - SSIM)
        loss_G_ssim = 1.0 - ssim_val 
        
        # TỔNG HỢP LOSS: 1 phần GAN + 100 phần L1 + 10 phần Cấu trúc
        g_loss = loss_G_bce + (100 * loss_G_l1) + (10 * loss_G_ssim)
        
        g_loss.backward()
        opt_g.step()
        
        log_message = f"Epoch {epoch} [{i}/{len(train_loader)}] D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f} SSIM_loss: {loss_G_ssim.item():.4f}"
        print(log_message)
        
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(log_message + "\n")

    checkpoint_data = {
        'epoch': epoch,
        'gen': gen.state_dict(),
        'disc': disc.state_dict(),     
        'opt_g': opt_g.state_dict(),   
        'opt_d': opt_d.state_dict()  
    }
    # Lưu dưới dạng gen_X.pth theo từng vòng
    torch.save(checkpoint_data, CHECKPOINT_DIR / f"gen_{epoch}.pth")
    # Lưu một bản copy "gen_last.pth" tiện cho việc resume tự động
    torch.save(checkpoint_data, CHECKPOINT_DIR / f"gen_last.pth")
    print(f"\n[SAVE] Đã lưu mô hình nâng cấp tại cuối Epoch {epoch}!\n")