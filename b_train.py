import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import StainingDataset
from pathlib import Path
from b_model import UNetGenerator, PatchDiscriminator

ROOT_DIR = Path(__file__).resolve().parent
DATA_ROOT = ROOT_DIR / "dataset" / "train"
CHECKPOINT_DIR = ROOT_DIR / "b_checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_FILE_PATH = ROOT_DIR / "b_log" / "train_log.txt"
LOG_FILE_PATH.parent.mkdir(exist_ok=True, parents=True)

# Data Loader
train_loader = DataLoader(StainingDataset(DATA_ROOT), batch_size=4, shuffle=True)

# Khởi tạo mạng ban đầu
gen = UNetGenerator().to(device)
disc = PatchDiscriminator().to(device)
opt_g = optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
opt_d = optim.Adam(disc.parameters(), lr=0.0002, betas=(0.5, 0.999))
criterion_gan = nn.BCEWithLogitsLoss()
criterion_l1 = nn.L1Loss()

# Resume training
start_epoch = 0
RESUME_CHECKPOINT = CHECKPOINT_DIR / "gen_6.pth" 

if RESUME_CHECKPOINT.exists():
    print(f"\n[RESUME] Đang nạp checkpoint: {RESUME_CHECKPOINT.name}...")
    checkpoint = torch.load(RESUME_CHECKPOINT, map_location=device)
    
    gen.load_state_dict(checkpoint['gen'])
    start_epoch = checkpoint['epoch'] + 1
    
    if 'disc' in checkpoint:
        disc.load_state_dict(checkpoint['disc'])
        print("-> Đã phục hồi trọng số Discriminator từ checkpoint.")
    else:
        print("-> Không tìm thấy Discriminator trong checkpoint.")
        
    if 'opt_g' in checkpoint:
        opt_g.load_state_dict(checkpoint['opt_g'])
    if 'opt_d' in checkpoint:
        opt_d.load_state_dict(checkpoint['opt_d'])
        
    print(f"[RESUME] Bắt đầu từ Epoch {start_epoch}\n")
else:
    print("\n[START] Bắt đầu train từ Epoch 0.\n")

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