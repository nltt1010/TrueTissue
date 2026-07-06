import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import StainingDataset
from pathlib import Path
from b_model import UNetGenerator, PatchDiscriminator

# --- CẤU HÌNH ---
DATA_ROOT = Path(r"G:/CV/pj/dataset/train")
CHECKPOINT_DIR = Path(r"G:/CV/pj/b_checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_FILE_PATH = "./b_log/train_log.txt"

# Data Loader
train_loader = DataLoader(StainingDataset(DATA_ROOT), batch_size=4, shuffle=True)

# Khởi tạo mạng ban đầu
gen = UNetGenerator().to(device)
disc = PatchDiscriminator().to(device)
opt_g = optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
opt_d = optim.Adam(disc.parameters(), lr=0.0002, betas=(0.5, 0.999))
criterion_gan = nn.BCEWithLogitsLoss()
criterion_l1 = nn.L1Loss()

# =====================================================================
# --- CƠ CHẾ RESUME TRAINING (TỰ ĐỘNG TRAIN TIẾP) ---
start_epoch = 0
# Chỉ định file checkpoint mới nhất bạn đang có (ở đây là gen_1.pth)
RESUME_CHECKPOINT = CHECKPOINT_DIR / "gen_6.pth" 

if RESUME_CHECKPOINT.exists():
    print(f"\n[RESUME] Tìm thấy file checkpoint cũ: {RESUME_CHECKPOINT.name}. Đang nạp dữ liệu...")
    checkpoint = torch.load(RESUME_CHECKPOINT, map_location=device)
    
    # 1. Phục hồi trọng số cho Generator
    gen.load_state_dict(checkpoint['gen'])
    
    # 2. Tính toán Epoch tiếp theo sẽ chạy (Epoch cũ + 1)
    start_epoch = checkpoint['epoch'] + 1
    
    # 3. Đọc thêm các thành phần khác nếu tồn tại (để tương thích với các checkpoint lưu trọn vẹn sau này)
    if 'disc' in checkpoint:
        disc.load_state_dict(checkpoint['disc'])
        print("-> Đã phục hồi trọng số Discriminator từ checkpoint.")
    else:
        print("-> Lưu ý: Checkpoint cũ 'gen_1.pth' không chứa Discriminator. Hệ thống sẽ để bộ phân biệt tự học lại từ đầu.")
        
    if 'opt_g' in checkpoint:
        opt_g.load_state_dict(checkpoint['opt_g'])
    if 'opt_d' in checkpoint:
        opt_d.load_state_dict(checkpoint['opt_d'])
        
    print(f"[RESUME] Nạp thành công! Tiến trình sẽ tiếp tục chạy từ Epoch {start_epoch}\n")
else:
    print("\n[VỪA CHẠY] Không tìm thấy checkpoint cũ hoặc cấu hình chạy mới. Bắt đầu từ Epoch 0.\n")
# =====================================================================

# Training loop
# Thay đổi cấu trúc lặp để bắt đầu từ start_epoch thay vì mặc định bằng 0
for epoch in range(start_epoch, 10):
    gen.train()
    for i, (real_in, real_out) in enumerate(train_loader):
        real_in, real_out = real_in.to(device), real_out.to(device)
        
        # Discriminator
        opt_d.zero_grad()
        fake_out = gen(real_in)
        d_real = disc(real_in, real_out)
        d_fake = disc(real_in, fake_out.detach())
        d_loss = criterion_gan(d_real, torch.ones_like(d_real)) + criterion_gan(d_fake, torch.zeros_like(d_fake))
        d_loss.backward()
        opt_d.step()
        
        # Generator
        opt_g.zero_grad()
        d_fake_for_g = disc(real_in, fake_out)
        g_loss = criterion_gan(d_fake_for_g, torch.ones_like(d_fake_for_g)) + 100 * criterion_l1(fake_out, real_out)
        g_loss.backward()
        opt_g.step()
        
        log_message = f"Epoch {epoch} [{i}/{len(train_loader)}] D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}"
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
    torch.save(checkpoint_data, CHECKPOINT_DIR / f"gen_{epoch}.pth")
    print(f"\n[SAVE] Đã lưu đầy đủ trạng thái hệ thống tại cuối Epoch {epoch}!\n")