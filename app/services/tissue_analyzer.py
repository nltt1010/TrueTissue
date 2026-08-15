import os
import sys
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# Thêm thư mục gốc vào sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from b_model import UNetGenerator as BaseUNet
from m_model import UNetGenerator as MyUNet
from classifier_model import ResNet18Classifier

GDRIVE_FOLDER_ID = "1_3JO8FZ0Gcbhlqv6-aU2TtYnOmBY1fiI"


def download_checkpoints_from_gdrive(root_dir=None, folder_id=GDRIVE_FOLDER_ID):
    """Tự động tải checkpoints từ Google Drive nếu chưa tồn tại trên máy / cloud."""
    if root_dir is None:
        root_dir = PROJECT_ROOT
    root_dir = Path(root_dir)
    
    b_dir = root_dir / "b_checkpoints"
    m_dir = root_dir / "m_checkpoints"
    cls_dir = root_dir / "cls_checkpoints"
    
    b_dir.mkdir(parents=True, exist_ok=True)
    m_dir.mkdir(parents=True, exist_ok=True)
    cls_dir.mkdir(parents=True, exist_ok=True)
    
    has_b = len(list(b_dir.glob("*.pth"))) > 0
    has_m = len(list(m_dir.glob("*.pth"))) > 0
    has_cls = len(list(cls_dir.glob("*.pth"))) > 0
    
    if has_b and has_m and has_cls:
        return True
        
    print(f"[TissueAnalyzer] Checkpoints missing. Downloading from Google Drive (Folder ID: {folder_id})...")
    try:
        import gdown
        import shutil
        
        # Tải folder từ Google Drive về root_dir
        gdown.download_folder(id=folder_id, output=str(root_dir), quiet=False, use_cookies=False)
        
        # Quét tìm và đồng bộ file .pth vào đúng thư mục checkpoints
        for pth in root_dir.rglob("*.pth"):
            parent_name = pth.parent.name.lower()
            pth_str = str(pth).lower()
            
            if "b_checkpoints" in parent_name or ("b_checkpoints" in pth_str and pth.parent != b_dir):
                target = b_dir / pth.name
                if not target.exists():
                    shutil.copy2(pth, target)
            elif "m_checkpoints" in parent_name or ("m_checkpoints" in pth_str and pth.parent != m_dir):
                target = m_dir / pth.name
                if not target.exists():
                    shutil.copy2(pth, target)
            elif "cls_checkpoints" in parent_name or ("cls_checkpoints" in pth_str and pth.parent != cls_dir):
                target = cls_dir / pth.name
                if not target.exists():
                    shutil.copy2(pth, target)
            elif "cls" in pth.name.lower() or "classifier" in pth.name.lower():
                target = cls_dir / pth.name
                if not target.exists():
                    shutil.copy2(pth, target)
                    
        print("[TissueAnalyzer] Checkpoints download and sync completed.")
        return True
    except Exception as e:
        print(f"[TissueAnalyzer] Warning: Could not download checkpoints from Google Drive: {e}")
        return False


class TissueAnalyzer:
    def __init__(self, root_dir=None):
        if root_dir is None:
            self.root_dir = PROJECT_ROOT
        else:
            self.root_dir = Path(root_dir)
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[TissueAnalyzer] Initialized on device: {self.device}")
        
        # Đường dẫn tới các thư mục checkpoint
        self.b_ckpt_dir = self.root_dir / "b_checkpoints"
        self.m_ckpt_dir = self.root_dir / "m_checkpoints"
        self.cls_ckpt_dir = self.root_dir / "cls_checkpoints"
        
        # Tự động tải checkpoint từ Google Drive nếu thiếu
        self.ensure_checkpoints()
        
        # Cache mô hình để tăng tốc
        self.model_cache = {}

    def ensure_checkpoints(self):
        """Đảm bảo các file checkpoint đã sẵn sàng, tự động tải từ Google Drive nếu thiếu."""
        return download_checkpoints_from_gdrive(self.root_dir)

    def get_available_checkpoints(self):
        """Trả về danh sách checkpoint .pth."""
        def list_pths(dir_path):
            if not dir_path.exists():
                return []
            pths = [f.name for f in dir_path.glob("*.pth")]
            # Sắp xếp tên file
            pths.sort()
            return pths

        ckpts = {
            "basemodel": list_pths(self.b_ckpt_dir),
            "mymodel": list_pths(self.m_ckpt_dir),
            "classifier": list_pths(self.cls_ckpt_dir)
        }
        
        # Nếu chưa có file nào, tự động tải và quét lại
        if not (ckpts["basemodel"] or ckpts["mymodel"] or ckpts["classifier"]):
            self.ensure_checkpoints()
            ckpts = {
                "basemodel": list_pths(self.b_ckpt_dir),
                "mymodel": list_pths(self.m_ckpt_dir),
                "classifier": list_pths(self.cls_ckpt_dir)
            }
        return ckpts

    def get_demo_samples(self):
        """Quét tìm các ảnh mẫu."""
        samples = []
        
        # 1. Kiểm tra ggtest/images.jpg
        gg_img = self.root_dir / "ggtest" / "images.jpg"
        if gg_img.exists():
            samples.append({
                "id": "ggtest_images",
                "name": "Demo 1: Standard Tissue Test (ggtest)",
                "path": str(gg_img),
                "category": "Unknown",
                "tissue": "General Tissue"
            })
            
        # 2. Quét dataset/test
        test_dir = self.root_dir / "dataset" / "test"
        if test_dir.exists():
            for cat in ["tumor", "normal"]:
                cat_dir = test_dir / cat
                if not cat_dir.exists():
                    continue
                for tissue_dir in cat_dir.iterdir():
                    if tissue_dir.is_dir() and tissue_dir.name.endswith("_grayscale"):
                        tissue_name = tissue_dir.name.replace("_grayscale", "").replace("_", " ").title()
                        # Lấy tối đa 2 ảnh cho mỗi loại
                        imgs = list(tissue_dir.glob("*.*"))
                        for i, img_p in enumerate(imgs[:2]):
                            if img_p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}:
                                samples.append({
                                    "id": f"test_{cat}_{tissue_dir.name}_{i}",
                                    "name": f"{cat.upper()}: {tissue_name} (Sample #{i+1})",
                                    "path": str(img_p),
                                    "category": cat.title(),
                                    "tissue": tissue_name
                                })
        return samples

    def _load_model_weights(self, model, ckpt_path):
        """Hỗ trợ load checkpoint."""
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
            
        ckpt = torch.load(ckpt_path, map_location=self.device)
        if isinstance(ckpt, dict):
            if "gen" in ckpt:
                model.load_state_dict(ckpt["gen"])
            elif "model_state" in ckpt:
                model.load_state_dict(ckpt["model_state"])
            elif "state_dict" in ckpt:
                model.load_state_dict(ckpt["state_dict"])
            else:
                try:
                    model.load_state_dict(ckpt)
                except Exception:
                    # Thử các key phổ biến
                    for k in ['model', 'backbone', 'net']:
                        if k in ckpt:
                            model.load_state_dict(ckpt[k])
                            break
        else:
            model.load_state_dict(ckpt)
        return model

    def get_stain_model(self, model_type="mymodel", ckpt_name=None):
        """Load và cache mô hình nhuộm màu."""
        model_type = model_type.lower()
        if model_type not in ["basemodel", "mymodel"]:
            raise ValueError(f"Invalid model_type: {model_type}. Must be 'basemodel' or 'mymodel'.")
            
        # Chọn checkpoint mặc định
        if not ckpt_name:
            avail = self.get_available_checkpoints()[model_type]
            if not avail:
                raise FileNotFoundError(f"No checkpoints found for {model_type}!")
            # Ưu tiên các checkpoint tốt nhất
            if "gen_8.pth" in avail and model_type == "mymodel":
                ckpt_name = "gen_8.pth"
            elif "gen_last.pth" in avail:
                ckpt_name = "gen_last.pth"
            elif "gen_6.pth" in avail and model_type == "basemodel":
                ckpt_name = "gen_6.pth"
            else:
                ckpt_name = avail[-1]
                
        cache_key = f"{model_type}_{ckpt_name}"
        if cache_key in self.model_cache:
            return self.model_cache[cache_key]
            
        ckpt_dir = self.b_ckpt_dir if model_type == "basemodel" else self.m_ckpt_dir
        ckpt_path = ckpt_dir / ckpt_name
        if not ckpt_path.exists():
            self.ensure_checkpoints()
        
        print(f"[TissueAnalyzer] Loading stain model ({model_type}): {ckpt_name}...")
        if model_type == "basemodel":
            model = BaseUNet().to(self.device)
        else:
            model = MyUNet().to(self.device)
            
        model = self._load_model_weights(model, ckpt_path)
        model.eval()
        self.model_cache[cache_key] = model
        return model

    def get_classifier_model(self, ckpt_name=None):
        """Load và cache mô hình phân loại."""
        if not ckpt_name:
            avail = self.get_available_checkpoints()["classifier"]
            if not avail:
                self.ensure_checkpoints()
                avail = self.get_available_checkpoints()["classifier"]
            if not avail:
                raise FileNotFoundError("No checkpoints found for classifier!")
            if "cls_best.pth" in avail:
                ckpt_name = "cls_best.pth"
            elif "cls_last.pth" in avail:
                ckpt_name = "cls_last.pth"
            else:
                ckpt_name = avail[-1]
                
        cache_key = f"classifier_{ckpt_name}"
        if cache_key in self.model_cache:
            return self.model_cache[cache_key]
            
        ckpt_path = self.cls_ckpt_dir / ckpt_name
        if not ckpt_path.exists():
            self.ensure_checkpoints()
            
        print(f"[TissueAnalyzer] Loading classifier model: {ckpt_name}...")
        
        model = ResNet18Classifier(num_classes=2, freeze_backbone=False).to(self.device)
        model = self._load_model_weights(model, ckpt_path)
        model.eval()
        self.model_cache[cache_key] = model
        return model

    def preprocess_for_staining(self, img_path_or_bgr, img_size=256):
        """Tiền xử lý ảnh xám."""
        if isinstance(img_path_or_bgr, (str, Path)):
            img = cv2.imread(str(img_path_or_bgr), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Could not read image from path: {img_path_or_bgr}")
        elif isinstance(img_path_or_bgr, np.ndarray):
            if img_path_or_bgr.ndim == 3:
                img = cv2.cvtColor(img_path_or_bgr, cv2.COLOR_BGR2GRAY)
            else:
                img = img_path_or_bgr.copy()
        else:
            raise TypeError("Input must be a file path or numpy ndarray.")
            
        img_resized = cv2.resize(img, (img_size, img_size))
        img_merged = cv2.merge([img_resized, img_resized, img_resized])
        img_norm = img_merged.astype(np.float32) / 127.5 - 1.0
        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        return img_tensor, img_resized

    def stain_image(self, img_input, model_type="mymodel", ckpt_name=None):
        """Chạy nhuộm màu ảo."""
        model = self.get_stain_model(model_type, ckpt_name)
        img_tensor, gray_resized = self.preprocess_for_staining(img_input)
        
        with torch.no_grad():
            fake_out = model(img_tensor)
            
        fake_out = fake_out.squeeze(0).cpu().numpy()
        fake_out = ((fake_out + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        fake_out = fake_out.transpose(1, 2, 0)
        # Chuyển đổi RGB sang BGR
        stained_bgr = cv2.cvtColor(fake_out, cv2.COLOR_RGB2BGR)
        return stained_bgr, gray_resized

    def predict_and_circle_abnormalities(self, stained_bgr, ckpt_name=None):
        """Phân loại và khoanh vùng bất thường."""
        model = self.get_classifier_model(ckpt_name)
        
        # Tiền xử lý ảnh cho ResNet
        img_rgb = cv2.cvtColor(stained_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        input_tensor = torch.FloatTensor(img_tensor).unsqueeze(0).to(self.device)
        input_tensor.requires_grad = True
        
        # Cài đặt hook cho Grad-CAM
        activations = []
        gradients = []
        
        target_layer = model.get_cam_target_layer()
        
        def forward_hook(module, input, output):
            activations.append(output)
            
        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])
            
        f_handle = target_layer.register_forward_hook(forward_hook)
        # Tương thích với Pytorch mới
        if hasattr(target_layer, 'register_full_backward_hook'):
            b_handle = target_layer.register_full_backward_hook(backward_hook)
        else:
            b_handle = target_layer.register_backward_hook(backward_hook)
            
        try:
            model.zero_grad()
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)[0]
            
            prob_normal = float(probs[0])
            prob_tumor = float(probs[1])
            is_abnormal = prob_tumor > prob_normal
            
            # Target class cho CAM: Class 1 (Tumor)
            score = logits[0, 1]
            score.backward()
            
            if activations and gradients:
                act = activations[0]
                grad = gradients[0]
                
                # Trọng số feature map
                weights = grad.mean(dim=(2, 3), keepdim=True)
                cam = (weights * act).sum(dim=1, keepdim=True)
                cam = torch.relu(cam)
                
                cam_img = cam.squeeze().cpu().detach().numpy()
                if cam_img.max() > 0:
                    cam_img = cam_img / cam_img.max()
            else:
                cam_img = np.zeros((256, 256), dtype=np.float32)
                
        finally:
            f_handle.remove()
            b_handle.remove()
            
        # Đổi kích thước heatmap
        h, w = stained_bgr.shape[:2]
        cam_resized = cv2.resize(cam_img, (w, h))
        
        # Dùng bản sao của ảnh đã nhuộm
        output_img = stained_bgr.copy()
        
        # Khoanh tròn các vùng bất thường
        num_circles = 0
        if is_abnormal:
            # Ngưỡng heatmap
            thresh_val = max(0.4, cam_resized.max() * 0.55)
            thresh = np.uint8((cam_resized > thresh_val) * 255)
            
            # Xử lý hình thái để làm sạch
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 45: # Lọc các đốm nhỏ
                    ((cx, cy), radius) = cv2.minEnclosingCircle(contour)
                    if radius > 6:
                        num_circles += 1
                        center = (int(cx), int(cy))
                        r = int(radius) + 4
                        
                        # 1. Vòng tròn đỏ bên ngoài
                        cv2.circle(output_img, center, r, (0, 0, 255), 2, cv2.LINE_AA)
                        # 2. Viền cyan bên trong
                        cv2.drawContours(output_img, [contour], -1, (0, 255, 255), 1, cv2.LINE_AA)
                        
                        # 3. Nhãn cho vòng tròn
                        label_text = f"Abnormal: {prob_tumor*100:.1f}%"
                        text_pos = (max(5, int(cx) - r), max(20, int(cy) - r - 6))
                        
                        # Nền chữ
                        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                        cv2.rectangle(output_img, (text_pos[0]-2, text_pos[1]-th-2), (text_pos[0]+tw+2, text_pos[1]+2), (0, 0, 0), -1)
                        cv2.putText(output_img, label_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 150, 255), 1, cv2.LINE_AA)
                        
        pred_dict = {
            "prediction": "Abnormal (Tumor)" if is_abnormal else "Normal (Healthy)",
            "is_abnormal": is_abnormal,
            "prob_normal": round(prob_normal * 100, 2),
            "prob_tumor": round(prob_tumor * 100, 2),
            "num_regions_detected": num_circles
        }
        return pred_dict, output_img

    def upscale_and_enhance(self, img_bgr, target_size=1024):
        """Nâng cấp ảnh lên 4x và làm sắc nét."""
        # 1. Phóng to ảnh
        upscaled = cv2.resize(img_bgr, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
        
        # 2. Tăng độ tương phản CLAHE
        lab = cv2.cvtColor(upscaled, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # 3. Làm sắc nét viền
        blurred = cv2.GaussianBlur(enhanced_bgr, (0, 0), 2.5)
        unsharp = cv2.addWeighted(enhanced_bgr, 2.2, blurred, -1.2, 0)
        
        # 4. Lọc sắc nét
        kernel = np.array([[0, -0.7, 0],
                           [-0.7, 3.8, -0.7],
                           [0, -0.7, 0]], dtype=np.float32)
        sharpened = cv2.filter2D(unsharp, -1, kernel)
        
        # 5. Tăng độ bão hòa màu
        hsv = cv2.cvtColor(sharpened, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.15, 0, 255)
        final_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        return final_bgr

    def run_full_analysis(self, img_input, model_type="mymodel", stain_ckpt=None, cls_ckpt=None, output_dir=None):
        """Chạy toàn bộ pipeline AI."""
        if output_dir is None:
            output_dir = self.root_dir / "app" / "static" / "outputs"
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Tiền tố ngẫu nhiên cho file
        import uuid
        run_id = str(uuid.uuid4())[:8]
        
        # Bước 1: Nhuộm màu
        stained_bgr, gray_img = self.stain_image(img_input, model_type, stain_ckpt)
        
        # Bước 2: Phân loại & Khoanh vùng
        pred_info, cam_bgr = self.predict_and_circle_abnormalities(stained_bgr, cls_ckpt)
        
        # Bước 3: Phóng to & Làm sắc nét
        upscaled_stained = self.upscale_and_enhance(stained_bgr, target_size=1024)
        upscaled_cam = self.upscale_and_enhance(cam_bgr, target_size=1024)
        upscaled_gray = cv2.resize(gray_img, (1024, 1024), interpolation=cv2.INTER_CUBIC)
        
        # Lưu các ảnh kết quả
        fn_gray = f"{run_id}_{model_type}_0_gray.png"
        fn_stain = f"{run_id}_{model_type}_1_stain.png"
        fn_cam = f"{run_id}_{model_type}_2_cam.png"
        fn_up_stain = f"{run_id}_{model_type}_3_up_stain.png"
        fn_up_cam = f"{run_id}_{model_type}_4_up_cam.png"
        
        cv2.imwrite(str(output_dir / fn_gray), gray_img)
        cv2.imwrite(str(output_dir / fn_stain), stained_bgr)
        cv2.imwrite(str(output_dir / fn_cam), cam_bgr)
        cv2.imwrite(str(output_dir / fn_up_stain), upscaled_stained)
        cv2.imwrite(str(output_dir / fn_up_cam), upscaled_cam)
        
        return {
            "run_id": run_id,
            "model_used": model_type,
            "stain_checkpoint": stain_ckpt or "default",
            "classifier_checkpoint": cls_ckpt or "default",
            "prediction": pred_info,
            "images": {
                "grayscale": f"/static/outputs/{fn_gray}",
                "stained": f"/static/outputs/{fn_stain}",
                "cam_circled": f"/static/outputs/{fn_cam}",
                "upscaled_stained": f"/static/outputs/{fn_up_stain}",
                "upscaled_cam": f"/static/outputs/{fn_up_cam}",
            }
        }

    def run_comparison(self, img_input, stain_ckpt_base=None, stain_ckpt_my=None, cls_ckpt=None, output_dir=None):
        """Chạy phân tích trên cả 2 mô hình để so sánh."""
        import uuid
        run_id = str(uuid.uuid4())[:8]
        if output_dir is None:
            output_dir = self.root_dir / "app" / "static" / "outputs"
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        res_base = self.run_full_analysis(img_input, "basemodel", stain_ckpt_base, cls_ckpt, output_dir)
        res_my = self.run_full_analysis(img_input, "mymodel", stain_ckpt_my, cls_ckpt, output_dir)
        
        return {
            "basemodel": res_base,
            "mymodel": res_my
        }

    def run_sliced_analysis(self, img_input, model_type="mymodel", stain_ckpt=None, cls_ckpt=None, output_dir=None):
        import math
        if isinstance(img_input, (str, Path)):
            img = cv2.imread(str(img_input))
        else:
            img = img_input.copy()
            
        H, W = img.shape[:2]
        # Bỏ qua cắt ảnh nếu nhỏ hơn 256x256
        if H <= 256 and W <= 256:
            return [self.run_full_analysis(img, model_type, stain_ckpt, cls_ckpt, output_dir)]
            
        N = max(1, math.ceil(H / 256))
        M = max(1, math.ceil(W / 256))
        step_h = H // N
        step_w = W // M
        
        results = []
        for i in range(N):
            for j in range(M):
                y_start = i * step_h
                y_end = H if i == N - 1 else (i + 1) * step_h
                x_start = j * step_w
                x_end = W if j == M - 1 else (j + 1) * step_w
                
                patch = img[y_start:y_end, x_start:x_end]
                patch_res = self.run_full_analysis(patch, model_type, stain_ckpt, cls_ckpt, output_dir)
                patch_res['patch_info'] = {'row': i, 'col': j, 'total_rows': N, 'total_cols': M, 'is_slice': True}
                results.append(patch_res)
        return results

    def run_sliced_comparison(self, img_input, stain_ckpt_base=None, stain_ckpt_my=None, cls_ckpt=None, output_dir=None):
        import math
        if isinstance(img_input, (str, Path)):
            img = cv2.imread(str(img_input))
        else:
            img = img_input.copy()
            
        H, W = img.shape[:2]
        # Bỏ qua cắt ảnh nếu nhỏ hơn 256x256
        if H <= 256 and W <= 256:
            return [self.run_comparison(img, stain_ckpt_base, stain_ckpt_my, cls_ckpt, output_dir)]
            
        N = max(1, math.ceil(H / 256))
        M = max(1, math.ceil(W / 256))
        step_h = H // N
        step_w = W // M
        
        results = []
        for i in range(N):
            for j in range(M):
                y_start = i * step_h
                y_end = H if i == N - 1 else (i + 1) * step_h
                x_start = j * step_w
                x_end = W if j == M - 1 else (j + 1) * step_w
                
                patch = img[y_start:y_end, x_start:x_end]
                patch_res = self.run_comparison(patch, stain_ckpt_base, stain_ckpt_my, cls_ckpt, output_dir)
                patch_res['patch_info'] = {'row': i, 'col': j, 'total_rows': N, 'total_cols': M, 'is_slice': True}
                results.append(patch_res)
        return results
