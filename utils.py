import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

class StainingDataset(Dataset):
    def __init__(self, root, img_size=256):
        self.img_size = img_size
        self.valid_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'}
        self.data_pairs = []

        # Quét tất cả file trong các thư mục *_grayscale
        # Cấu trúc: dataset/train/normal/tissue_name_grayscale/
        all_gray_files = list(Path(root).rglob("*_grayscale/*"))
        
        for gray_path in all_gray_files:
            if gray_path.suffix.lower() not in self.valid_extensions:
                continue
                
            # Suy luận đường dẫn target (ảnh màu gốc)
            # parent_dir là .../tissue_name_grayscale/
            parent_dir = gray_path.parent
            # target_dir là .../tissue_name/
            target_dir = parent_dir.parent / parent_dir.name.replace("_grayscale", "")
            target_path = target_dir / gray_path.name
            
            if target_path.exists():
                self.data_pairs.append((gray_path, target_path))
        
        print(f"Dataset đã tải xong: {len(self.data_pairs)} cặp file.")
        
    def __len__(self): 
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        gray_path, target_path = self.data_pairs[idx]
        
        img_gray = cv2.imread(str(gray_path), cv2.IMREAD_GRAYSCALE)
        img_color = cv2.imread(str(target_path))
        
        if img_gray is None or img_color is None:
            return self.__getitem__((idx + 1) % len(self))
            
        # Resize đồng nhất
        img_gray = cv2.resize(img_gray, (self.img_size, self.img_size))
        img_color = cv2.resize(img_color, (self.img_size, self.img_size))
        
        # Tiền xử lý
        img_gray = cv2.merge([img_gray, img_gray, img_gray]) 
        img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
        
        # Chuẩn hóa về [-1, 1]
        img_gray = (img_gray.transpose(2,0,1).astype(np.float32) / 127.5) - 1
        img_color = (img_color.transpose(2,0,1).astype(np.float32) / 127.5) - 1
        
        return torch.FloatTensor(img_gray), torch.FloatTensor(img_color)