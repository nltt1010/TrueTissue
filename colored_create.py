import cv2
import torch
import numpy as np
from pathlib import Path
# Import mô hình
from m_model import UNetGenerator

BASE_DIR = Path(__file__).resolve().parent
DATA_ROOTS = [
    BASE_DIR / "dataset" / "train",
    BASE_DIR / "dataset" / "test"
]

BEST_CHECKPOINT_PATH = BASE_DIR / "m_checkpoints" / "gen_8.pth"

IMG_SIZE = 256
VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_fake_dataset():
    if not BEST_CHECKPOINT_PATH.exists():
        print(f"[LỖI] Không tìm thấy file checkpoint tốt nhất tại: {BEST_CHECKPOINT_PATH}")
        return

    print(f"[CẤU HÌNH] Đang sử dụng thiết bị: {device}")
    print(f"[LOAD MODEL] Đang nạp mô hình tốt nhất: {BEST_CHECKPOINT_PATH.name}...")
    
    # Khởi tạo mô hình
    gen = UNetGenerator().to(device)
    checkpoint = torch.load(BEST_CHECKPOINT_PATH, map_location=device)
    gen.load_state_dict(checkpoint['gen'])
    gen.eval()
    print("-> Đã nạp thành công trọng số .pth file!\n")

    total_processed = 0

    # Duyệt qua dataset
    for data_root in DATA_ROOTS:
        if not data_root.exists():
            print(f"[BỎ QUA] Đường dẫn không tồn tại: {data_root}")
            continue

        print(f"==================== ĐANG XỬ LÝ: {data_root} ====================")
        
        # Quét thư mục ảnh xám
        all_gray_files = [
            p for p in data_root.rglob("*_grayscale/*") 
            if p.suffix.lower() in VALID_EXTENSIONS
        ]
        
        print(f"Tìm thấy {len(all_gray_files)} ảnh xám cần nhuộm màu giả lập...")

        with torch.no_grad():
            for i, gray_path in enumerate(all_gray_files):
                # Tạo thư mục đích
                gray_dir = gray_path.parent
                fake_colored_dir_name = gray_dir.name.replace("_grayscale", "_fake_colored")
                fake_colored_dir = gray_dir.parent / fake_colored_dir_name
                fake_colored_dir.mkdir(exist_ok=True, parents=True)

                target_fake_path = fake_colored_dir / gray_path.name

                img_gray = cv2.imread(str(gray_path), cv2.IMREAD_GRAYSCALE)
                if img_gray is None:
                    continue

                img_gray_resized = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE))
                
                img_gray_3ch = cv2.merge([img_gray_resized, img_gray_resized, img_gray_resized])
                tensor_in = (img_gray_3ch.transpose(2, 0, 1).astype(np.float32) / 127.5) - 1.0
                tensor_in = torch.FloatTensor(tensor_in).unsqueeze(0).to(device)

                fake_out = gen(tensor_in)

                img_fake_np = ((fake_out.squeeze(0).cpu().numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8)
                img_fake_bgr = cv2.cvtColor(img_fake_np, cv2.COLOR_RGB2BGR)

                cv2.imwrite(str(target_fake_path), img_fake_bgr)
                total_processed += 1

                if (i + 1) % 50 == 0 or (i + 1) == len(all_gray_files):
                    print(f"    Tiến độ: [{i+1}/{len(all_gray_files)}] ảnh đã được nhuộm và lưu.")

    print("=" * 65)
    print(f"✅ ĐÃ HOÀN THÀNH! Tổng cộng đã sinh {total_processed} ảnh màu giả lập (Fake Colored).")
    print("📁 Dữ liệu mới đã nằm trong các thư mục '*_fake_colored' cùng cấp với dataset của bạn.")

if __name__ == "__main__":
    generate_fake_dataset()