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

        # Quét thư mục *_grayscale
        all_gray_files = list(Path(root).rglob("*_grayscale/*"))
        
        for gray_path in all_gray_files:
            if gray_path.suffix.lower() not in self.valid_extensions:
                continue
                
            # Tìm đường dẫn ảnh gốc
            parent_dir = gray_path.parent
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
            
        # Resize ảnh
        img_gray = cv2.resize(img_gray, (self.img_size, self.img_size))
        img_color = cv2.resize(img_color, (self.img_size, self.img_size))
        
        # Chuẩn hóa định dạng
        img_gray = cv2.merge([img_gray, img_gray, img_gray]) 
        img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
        
        # Chuẩn hóa [-1, 1]
        img_gray = (img_gray.transpose(2,0,1).astype(np.float32) / 127.5) - 1
        img_color = (img_color.transpose(2,0,1).astype(np.float32) / 127.5) - 1
        
        return torch.FloatTensor(img_gray), torch.FloatTensor(img_color)



class ClassificationDataset(Dataset):
    def __init__(self, root, img_size=256, mode='fake'):
        """
        Khởi tạo dataset cho phân loại
        """
        self.img_size = img_size
        self.valid_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'}
        self.samples = []
        
        root_path = Path(root)

        # Lấy danh sách nhãn
        class_dirs = [d for d in root_path.iterdir() if d.is_dir()]
        class_dirs.sort()
        
        self.class_to_idx = {cls_dir.name: i for i, cls_dir in enumerate(class_dirs)}
        self.num_classes = len(self.class_to_idx)
        print(f"Đã tìm thấy các lớp phân loại: {self.class_to_idx}")

        for cls_name, cls_idx in self.class_to_idx.items():
            cls_folder = root_path / cls_name
            
            for img_path in cls_folder.rglob("*"):
                if img_path.suffix.lower() in self.valid_extensions:
                    parent_name = str(img_path.parent)
                    
                    if mode == 'fake':
                        # Lấy ảnh giả lập
                        if "_fake_colored" in parent_name:
                            self.samples.append((img_path, cls_idx))
                    elif mode == 'real':
                        # Lấy ảnh gốc
                        if "_grayscale" not in parent_name and "_fake_colored" not in parent_name:
                            self.samples.append((img_path, cls_idx))
                            
        print(f"ClassificationDataset ({mode.upper()} MODE) đã tải xong: {len(self.samples)} mẫu ảnh.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Đọc ảnh màu
        img_color = cv2.imread(str(img_path))
        if img_color is None:
            return self.__getitem__((idx + 1) % len(self))
            
        img_color = cv2.resize(img_color, (self.img_size, self.img_size))
        img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
        
        # Chuẩn hóa [0, 1]
        img_tensor = img_color.transpose(2, 0, 1).astype(np.float32) / 255.0
        
        return torch.FloatTensor(img_tensor), torch.tensor(label, dtype=torch.long)