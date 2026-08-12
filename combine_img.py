import os
import random
import math
from PIL import Image

def get_grid_dimensions(num_images):
    """
    Tính kích thước lưới tối ưu
    """
    sqrt_num = math.isqrt(num_images)
    if sqrt_num * sqrt_num == num_images:
        return sqrt_num, sqrt_num
    
    # Tìm ước số cân đối
    best_rows, best_cols = 1, num_images
    min_diff = num_images
    
    for i in range(1, sqrt_num + 1):
        if num_images % i == 0:
            rows = i
            cols = num_images // i
            if (cols - rows) < min_diff:
                min_diff = cols - rows
                best_rows, best_cols = rows, cols
                
    return best_rows, best_cols

def suggest_image_counts(total_available):
    """
    Gợi ý số lượng ảnh
    """
    suggestions = []
    for count in range(1, total_available + 1):
        rows, cols = get_grid_dimensions(count)
        if count in [4, 6, 8, 9, 10, 12, 15, 16, 20, 24, 25, 30, 36, 48, 50, 64] or (rows > 1 and cols > 1):
            suggestions.append((count, f"{rows}x{cols} ({rows} hàng x {cols} cột)"))
    return suggestions

def combine_images_grid(folder_path, output_path="grid_result.jpg", num_images=4):
    """
    Ghép ảnh thành lưới
    """
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')
    
    all_files = [
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if f.lower().endswith(valid_extensions)
    ]
    
    total_available = len(all_files)
    if total_available == 0:
        print(f"Không tìm thấy ảnh nào trong thư mục: {folder_path}")
        return

    print(f"\n==========================================")
    print(f"TỔNG SỐ ẢNH TÌM THẤY TRONG THƯ MỤC: {total_available}")
    
    # In gợi ý
    suggestions = suggest_image_counts(total_available)
    print("\n[GỢI Ý SỐ LƯỢNG ẢNH ĐỂ GHÉP LƯỚI ĐẸP]:")
    for count, grid_str in suggestions:
        print(f" - Ghép {count} ảnh  -> Bố cục lưới: {grid_str}")
    print("==========================================\n")

    if num_images > total_available:
        print(f"Cảnh báo: Bạn chọn {num_images} ảnh nhưng thư mục chỉ có {total_available} ảnh. Sẽ dùng {total_available} ảnh.")
        num_images = total_available

    rows, cols = get_grid_dimensions(num_images)
    actual_num_images = rows * cols

    selected_files = random.sample(all_files, actual_num_images)
    images = [Image.open(f) for f in selected_files]

    # Cố định kích thước ảnh
    tile_width = images[0].width
    tile_height = images[0].height

    canvas_width = tile_width * cols
    canvas_height = tile_height * rows

    print(f"Đã chọn {actual_num_images} ảnh ngẫu nhiên.")
    print(f"Kích thước mỗi ô ảnh đơn: {tile_width} x {tile_height} px")
    print(f"Bố cục ghép: {rows} hàng x {cols} cột")
    print(f"--> KÍCH THƯỚC ẢNH KẾT QUẢ ĐẦU RA: {canvas_width} x {canvas_height} px\n")

    combined_image = Image.new("RGB", (canvas_width, canvas_height))

    # Ghép ảnh vào lưới
    for idx, img in enumerate(images):
        if img.size != (tile_width, tile_height):
            img = img.resize((tile_width, tile_height), Image.Resampling.LANCZOS)
            
        r = idx // cols
        c = idx % cols
        
        x_pos = c * tile_width
        y_pos = r * tile_height
        
        combined_image.paste(img, (x_pos, y_pos))

    # Lưu kết quả
    combined_image.save(output_path)
    print(f"GHÉP THÀNH CÔNG! File được lưu tại: {output_path}")


if __name__ == "__main__":
    from pathlib import Path
    BASE_DIR = Path(__file__).resolve().parent
    FOLDER = BASE_DIR / "dataset" / "test" / "tumor" / "digestive_tissue_grayscale"
    combine_images_grid(folder_path=str(FOLDER), output_path="ket_qua_ghep_luoi.jpg", num_images=4)