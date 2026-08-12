import cv2
import numpy as np
from pathlib import Path
import albumentations as A

class DatasetProcessor:
    def __init__(self, dataset_root):
        self.dataset_root = Path(dataset_root)
        
    def generate_grayscale(self):
        for split in ['train', 'test']:
            split_path = self.dataset_root / split
            if not split_path.exists(): continue
            
            for cat in ['normal', 'tumor']:
                cat_path = split_path / cat
                if not cat_path.exists(): continue
                
                for tissue_dir in cat_path.iterdir():
                    if tissue_dir.is_dir() and not tissue_dir.name.endswith('_grayscale'):
                        self._process_tissue(tissue_dir)

    def _process_tissue(self, tissue_dir):
        gray_dir = tissue_dir.parent / f"{tissue_dir.name}_grayscale"
        gray_dir.mkdir(parents=True, exist_ok=True)
        print(f"Đang xử lý: {tissue_dir.name}")
        
        for img_path in tissue_dir.iterdir():
            if img_path.is_file() and img_path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}:
                dest_path = gray_dir / img_path.name
                if dest_path.exists(): continue
                
                img = cv2.imread(str(img_path))
                if img is not None:
                    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(str(dest_path), gray_img)

if __name__ == "__main__":
    DATASET_PATH = Path(__file__).resolve().parent / "dataset"
    processor = DatasetProcessor(DATASET_PATH)
    processor.generate_grayscale()
    print("Hoàn tất tạo ảnh Grayscale cho toàn bộ dataset.")