import torch
import cv2
import numpy as np
from pathlib import Path
from b_model import UNetGenerator

# TEST_DATA_ROOT should point to a single grayscale image file for b_test_single
TEST_DATA_ROOT = Path(r"G:/CV/pj/ggtest/images.jpg")
CHECKPOINT_DIR = Path(r"G:/CV/pj/checkpoints")
BASE_OUTPUT_DIR = Path(r"G:/CV/pj/evaluation_resultsingle")

# =====================================================================
LIST_CHECKPOINTS = [
    "gen_0.pth",
    "gen_1.pth",
    "gen_2.pth"
]
# =====================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_image(image_path: Path, img_size: int = 256) -> torch.Tensor:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Không thể đọc ảnh: {image_path}")

    image = cv2.resize(image, (img_size, img_size))
    image = cv2.merge([image, image, image])
    image = image.astype(np.float32) / 127.5 - 1.0
    image = image.transpose(2, 0, 1)
    return torch.from_numpy(image).unsqueeze(0)


def save_comparison(input_gray: np.ndarray, output_color: np.ndarray, save_path: Path):
    input_gray = input_gray.astype(np.uint8)
    if input_gray.ndim == 2:
        input_gray = cv2.cvtColor(input_gray, cv2.COLOR_GRAY2BGR)
    comparison = np.hstack((input_gray, output_color))
    cv2.imwrite(str(save_path), comparison)


def main_evaluation():
    if not TEST_DATA_ROOT.exists():
        print(f"[LỖI] Không tìm thấy ảnh test tại: {TEST_DATA_ROOT}")
        return
    if not TEST_DATA_ROOT.is_file():
        print(f"[LỖI] TEST_DATA_ROOT phải là một file ảnh, không phải thư mục: {TEST_DATA_ROOT}")
        return

    BASE_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    input_tensor = preprocess_image(TEST_DATA_ROOT, img_size=256).to(device)
    input_gray = cv2.imread(str(TEST_DATA_ROOT), cv2.IMREAD_GRAYSCALE)
    input_gray = cv2.resize(input_gray, (256, 256))

    print(f"Tìm thấy {len(LIST_CHECKPOINTS)} mô hình trong danh sách để đánh giá.")

    gen = UNetGenerator().to(device)

    for ckpt_name in LIST_CHECKPOINTS:
        ckpt_path = CHECKPOINT_DIR / ckpt_name
        if not ckpt_path.exists():
            print(f"[BỎ QUA] Không tìm thấy file checkpoint: {ckpt_name}")
            continue

        model_id = ckpt_path.stem
        print(f"==================== ĐANG ĐÁNH GIÁ MODEL: {model_id} ====================")

        model_output_dir = BASE_OUTPUT_DIR / f"test_results_{model_id}"
        model_output_dir.mkdir(exist_ok=True, parents=True)

        checkpoint = torch.load(ckpt_path, map_location=device)
        gen.load_state_dict(checkpoint["gen"])
        gen.eval()

        with torch.no_grad():
            fake_out = gen(input_tensor)

        fake_out = fake_out.squeeze(0).cpu().numpy()
        fake_out = ((fake_out + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        fake_out = fake_out.transpose(1, 2, 0)
        fake_out = cv2.cvtColor(fake_out, cv2.COLOR_RGB2BGR)

        output_path = model_output_dir / f"{model_id}_output.png"
        save_comparison(input_gray, fake_out, output_path)

        print(f"-> Lưu kết quả ảnh tại: {output_path}")


if __name__ == "__main__":
    main_evaluation()
