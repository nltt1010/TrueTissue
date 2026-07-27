import torch
import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from utils import StainingDataset
from b_model import UNetGenerator

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

TEST_DATA_ROOT = Path(r"G:/CV/pj/dataset/test")

CHECKPOINT_DIR = Path(r"G:/CV/pj/b_checkpoints")
BASE_OUTPUT_DIR = Path(r"G:/CV/pj/b_evaluation_results")

LOG_DIR = Path("./b_log_vstain")

# =====================================================================
LIST_CHECKPOINTS = [
    "gen_0.pth",
    "gen_1.pth",
    "gen_2.pth",
    "gen_3.pth",
    "gen_4.pth",
    "gen_5.pth",
    "gen_6.pth",
    "gen_7.pth",
    "gen_8.pth",
    "gen_9.pth",
]

MAX_TEST_IMAGES = 100
# =====================================================================

# Đảm bảo các thư mục tồn tại an toàn trước khi ghi file
BASE_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
LOG_DIR.mkdir(exist_ok=True, parents=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main_evaluation():
    if not TEST_DATA_ROOT.exists():
        print(f"[LỖI] Không tìm thấy thư mục dữ liệu test tại: {TEST_DATA_ROOT}")
        return
    test_loader = DataLoader(StainingDataset(TEST_DATA_ROOT), batch_size=1, shuffle=False)
    
    total_images = min(len(test_loader), MAX_TEST_IMAGES)
    
    gen = UNetGenerator().to(device)
    global_summary_results = []

    print(f"Tìm thấy {len(LIST_CHECKPOINTS)} mô hình trong danh sách để đánh giá.")
    print(f"CHẾ ĐỘ TEST NHANH: Chỉ chạy đúng {total_images} ảnh đầu tiên của tập dữ liệu.\n")

    for ckpt_name in LIST_CHECKPOINTS:
        ckpt_path = CHECKPOINT_DIR / ckpt_name
        if not ckpt_path.exists():
            print(f"[BỎ QUA] Không tìm thấy file checkpoint: {ckpt_name}")
            continue
            
        model_id = ckpt_path.stem  
        print(f"==================== ĐANG ĐÁNH GIÁ MODEL: {model_id} ====================")
        
        # Thư mục lưu ảnh kết quả trực quan (Giữ nguyên trong folder kết quả chính)
        model_output_dir = BASE_OUTPUT_DIR / f"test_results_{model_id}"
        model_output_dir.mkdir(exist_ok=True, parents=True)
        
        # File .txt của từng model sẽ được lưu vào thư mục ./log bên ngoài
        per_image_file_path = LOG_DIR / f"eval_{model_id}.txt"
        
        checkpoint = torch.load(ckpt_path, map_location=device)
        gen.load_state_dict(checkpoint['gen'])
        gen.eval()

        psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        fid_metric = FrechetInceptionDistance(feature=64, normalize=False).to(device)
        lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(device)

        with open(per_image_file_path, "w", encoding="utf-8") as f_per:
            f_per.write(f"=== KẾT QUẢ ĐÁNH GIÁ CHI TIẾT (TEST NHANH 30 ẢNH) - MODEL {model_id} ===\n")
            f_per.write(f"STT\t\tPSNR (dB)\tSSIM\t\tLPIPS\n")
            f_per.write("-" * 55 + "\n")
            
            with torch.no_grad():
                for i, (real_in, real_out) in enumerate(test_loader):
                    if i >= MAX_TEST_IMAGES:
                        break
                        
                    real_in, real_out = real_in.to(device), real_out.to(device)
                    fake_out = gen(real_in)
                    
                    fake_01 = (fake_out + 1) / 2.0
                    real_01 = (real_out + 1) / 2.0
                    
                    p_img = psnr_metric(fake_01, real_01).item()
                    s_img = ssim_metric(fake_01, real_01).item()
                    l_img = lpips_metric(fake_out, real_out).item()
                    
                    f_per.write(f"Ảnh_{i:04d}\t{p_img:.4f}\t\t{s_img:.4f}\t\t{l_img:.4f}\n")
                    
                    psnr_metric.update(fake_01, real_01)
                    ssim_metric.update(fake_01, real_01)
                    lpips_metric.update(fake_out, real_out)
                    
                    fake_uint8 = ((fake_out + 1) * 127.5).clamp(0, 255).byte()
                    real_uint8 = ((real_out + 1) * 127.5).clamp(0, 255).byte()
                    fid_metric.update(real_uint8, real=True)
                    fid_metric.update(fake_uint8, real=False)

                    # Lưu ảnh trực quan đối chiếu
                    img_gray = ((real_in.squeeze(0).cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)
                    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_RGB2BGR)
                    
                    img_fake = ((fake_out.squeeze(0).cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)
                    img_fake = cv2.cvtColor(img_fake, cv2.COLOR_RGB2BGR)
                    
                    img_real = ((real_out.squeeze(0).cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)
                    img_real = cv2.cvtColor(img_real, cv2.COLOR_RGB2BGR)

                    comparison_img = np.hstack((img_gray, img_fake, img_real))
                    
                    cv2.imwrite(str(model_output_dir / f"img_{i:04d}.png"), comparison_img)

                    print(f"   [Tiến độ] Đã xử lý ảnh: {i + 1}/{total_images}...")

        avg_psnr = psnr_metric.compute().item()
        avg_ssim = ssim_metric.compute().item()
        avg_lpips = lpips_metric.compute().item()
        final_fid = fid_metric.compute().item()
        
        print(f"-> Hoàn thành {model_id}! Log chi tiết tại: ./log/eval_{model_id}.txt\n")
        
        global_summary_results.append({
            'model': model_id,
            'psnr': avg_psnr,
            'ssim': avg_ssim,
            'fid': final_fid,
            'lpips': avg_lpips
        })

    # File tổng hợp kết quả so sánh cũng sẽ xuất thẳng vào thư mục ./log bên ngoài
    summary_total_path = LOG_DIR / "eval_summary_total.txt"
    with open(summary_total_path, "w", encoding="utf-8") as f_sum:
        f_sum.write("=" * 75 + "\n")
        f_sum.write(" BẢNG TỔNG HỢP VÀ SO SÁNH CHỈ SỐ BASEMODEL ".center(75, " ") + "\n")
        f_sum.write("=" * 75 + "\n")
        f_sum.write(f"{'TÊN MODEL':<15}\t{'PSNR (↑)':<12}\t{'SSIM (↑)':<12}\t{'FID (↓)':<12}\t{'LPIPS (↓)':<12}\n")
        f_sum.write("-" * 75 + "\n")
        for res in global_summary_results:
            f_sum.write(f"{res['model']:<15}\t{res['psnr']:<12.4f}\t{res['ssim']:<12.4f}\t{res['fid']:<12.4f}\t{res['lpips']:<12.4f}\n")
        f_sum.write("=" * 75 + "\n")
        f_sum.write("Ghi chú: (↑) Càng cao càng tốt | (↓) Càng thấp càng tốt.\n")
        f_sum.write("Lưu ý: Chỉ số FID tính toán trên dữ liệu ít (30 ảnh) chỉ mang tính chất kiểm tra luồng code.\n")

    print(f"\n[XUẤT FILE THÀNH CÔNG] Đã tạo bảng so sánh tổng hợp tại: {summary_total_path}")

if __name__ == "__main__":
    main_evaluation()