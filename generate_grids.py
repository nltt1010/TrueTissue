import cv2
import torch
import numpy as np
from pathlib import Path
import random
import os

from app.services.tissue_analyzer import TissueAnalyzer

def create_grid_image(images, rows=3, cols=4, col_headers=None, row_headers=None, cell_size=256):
    """
    Tạo ảnh lưới chứa danh sách ảnh
    """
    header_h = 40 if col_headers else 0
    row_header_w = 120 if row_headers else 0
    
    canvas_h = header_h + (rows * cell_size)
    canvas_w = row_header_w + (cols * cell_size)
    
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
    
    if col_headers:
        for j, text in enumerate(col_headers):
            x = row_header_w + j * cell_size
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = x + (cell_size - text_size[0]) // 2
            text_y = int(header_h * 0.7)
            cv2.putText(canvas, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
    if row_headers:
        for i, text in enumerate(row_headers):
            y = header_h + i * cell_size
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            text_x = (row_header_w - text_size[0]) // 2
            text_y = y + (cell_size + text_size[1]) // 2
            cv2.putText(canvas, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
    for idx, img in enumerate(images):
        if img is None:
            continue
        i = idx // cols
        j = idx % cols
        
        y = header_h + i * cell_size
        x = row_header_w + j * cell_size
        
        img_resized = cv2.resize(img, (cell_size, cell_size))
        canvas[y:y+cell_size, x:x+cell_size] = img_resized
        cv2.rectangle(canvas, (x, y), (x+cell_size, y+cell_size), (200, 200, 200), 1)
        
    return canvas

def main():
    print("Starting Grid Generation Script...")
    analyzer = TissueAnalyzer()
    
    output_dir = Path("grid_results")
    output_dir.mkdir(exist_ok=True)
    
    base_dir = Path(__file__).resolve().parent / "dataset" / "test"
    tissue_types = ["digestive", "lymph_node", "prostate"]
    classes = ["normal", "tumor"]
    
    print("Loading models...")
    analyzer.get_stain_model("basemodel")
    analyzer.get_stain_model("mymodel")
    analyzer.get_classifier_model()
    
    col_headers = ["Ground Truth", "Grayscale", "Staining", "Predict"]
    row_headers = ["Digestive", "Lymph Node", "Prostate"]
    
    for cls in classes:
        print(f"\nProcessing class: {cls}")
        
        selected_images = {}
        for tissue in tissue_types:
            folder_gt = base_dir / cls / f"{tissue}_tissue"
            folder_gray = base_dir / cls / f"{tissue}_tissue_grayscale"
            
            if not folder_gt.exists() or not folder_gray.exists():
                print(f"Skipping {tissue} - folders not found.")
                continue
                
            gt_files = list(folder_gt.glob("*.png")) + list(folder_gt.glob("*.jpg"))
            if not gt_files:
                continue
                
            chosen_gt = random.choice(gt_files)
            chosen_gray = folder_gray / chosen_gt.name
            
            selected_images[tissue] = {
                "gt": cv2.imread(str(chosen_gt)),
                "gray": cv2.imread(str(chosen_gray))
            }
        
        if not selected_images:
            continue
            
        for model_config in ["basemodel", "mymodel", "real_clf"]:
            print(f"  Generating grid for: {model_config}")
            grid_images = []
            
            for tissue in tissue_types:
                if tissue not in selected_images:
                    grid_images.extend([None, None, None, None])
                    continue
                    
                gt_img = selected_images[tissue]["gt"]
                gray_img = selected_images[tissue]["gray"]
                
                if model_config in ["basemodel", "mymodel"]:
                    stained_bgr, _ = analyzer.stain_image(gray_img, model_type=model_config)
                    _, predict_bgr = analyzer.predict_and_circle_abnormalities(stained_bgr)
                else:
                    stained_bgr = gt_img.copy()
                    _, predict_bgr = analyzer.predict_and_circle_abnormalities(stained_bgr)
                    
                grid_images.append(gt_img)
                grid_images.append(gray_img)
                grid_images.append(stained_bgr)
                grid_images.append(predict_bgr)
                
            grid_canvas = create_grid_image(grid_images, rows=3, cols=4, col_headers=col_headers, row_headers=row_headers)
            
            out_name = f"Grid_{model_config}_{cls}.png"
            cv2.imwrite(str(output_dir / out_name), grid_canvas)
            print(f"    Saved {out_name}")

    print("\n[DONE] All 6 grids have been generated in the 'grid_results' folder!")

if __name__ == "__main__":
    main()
