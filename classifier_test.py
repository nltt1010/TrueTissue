import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Import mô hình và dataset
from classifier_model import ResNet18Classifier
from utils import ClassificationDataset

ROOT_DIR = Path(__file__).resolve().parent
TEST_DATA_ROOT = ROOT_DIR / "dataset" / "test"
CHECKPOINT_DIR = ROOT_DIR / "cls_checkpoints"

LOG_DIR = ROOT_DIR / "cls_log"
LOG_DIR.mkdir(exist_ok=True, parents=True)
TEST_LOG_FILE = LOG_DIR / "classifier_test_all_epochs_results.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16

def test_all_checkpoints():
    print(f"\n[BẮT ĐẦU] Đánh giá ResNet18 Classifier qua TẤT CẢ các Epoch trên tập TEST...")
    print(f"Thiết bị: {device}")
    
    if not TEST_DATA_ROOT.exists():
        print(f"[LỖI] Không tìm thấy thư mục Test: {TEST_DATA_ROOT}")
        return
        
    if not CHECKPOINT_DIR.exists():
        print(f"[LỖI] Không tìm thấy thư mục chứa checkpoints: {CHECKPOINT_DIR}")
        return

    # Lọc danh sách checkpoint
    checkpoint_files = []
    for f in CHECKPOINT_DIR.glob("cls_*.pth"):
        num_str = f.stem.replace("cls_", "")
        if num_str.isdigit():
            checkpoint_files.append((int(num_str), f))
            
    # Sắp xếp theo thứ tự epoch tăng dần (0, 1, 2, ..., 9)
    checkpoint_files.sort(key=lambda x: x[0])
    
    if not checkpoint_files:
        print("[LỖI] Không tìm thấy file checkpoint nào theo định dạng cls_X.pth")
        return

    print(f"\nĐang load dataset test...")
    test_dataset = ClassificationDataset(TEST_DATA_ROOT, img_size=256, mode='fake')
    
    if len(test_dataset) == 0:
        print("[LỖI] Tập test trống. Không có ảnh nào để đánh giá.")
        return
        
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    num_classes = getattr(test_dataset, 'num_classes', 2)

    model = ResNet18Classifier(num_classes=num_classes, freeze_backbone=False).to(device)
    
    # Mở file log
    with open(TEST_LOG_FILE, "w", encoding="utf-8") as f:
        f.write("=========================================================================================\n")
        f.write("      KẾT QUẢ KIỂM THỬ RESNET-18 QUA TỪNG EPOCH (TẬP TEST: {} ẢNH)\n".format(len(test_dataset)))
        f.write("=========================================================================================\n")
        f.write(f"{'Epoch':<7} | {'Accuracy':<9} | {'Sensitivity':<11} | {'Specificity':<11} | {'Precision':<9} | {'F1-Score':<9} | {'ROC-AUC':<8}\n")
        f.write("-" * 89 + "\n")

    print("\n" + "="*89)
    print(f"{'Epoch':<7} | {'Accuracy':<9} | {'Sensitivity':<11} | {'Specificity':<11} | {'Precision':<9} | {'F1-Score':<9} | {'ROC-AUC':<8}")
    print("-" * 89)

    for epoch_num, ckpt_path in checkpoint_files:
        # Nạp trọng số
        checkpoint = torch.load(ckpt_path, map_location=device)
        if 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
        else:
            model.load_state_dict(checkpoint)
            
        model.eval()

        all_labels = []
        all_preds = []
        all_probs = []
        
        # Chạy dự đoán
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.cpu().numpy()
                
                outputs = model(inputs)
                
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                _, preds = torch.max(outputs, 1)
                preds = preds.cpu().numpy()
                
                all_labels.extend(labels)
                all_preds.extend(preds)
                all_probs.extend(probs)

        accuracy = accuracy_score(all_labels, all_preds) * 100.0
        precision = precision_score(all_labels, all_preds, zero_division=0) * 100.0
        sensitivity = recall_score(all_labels, all_preds, zero_division=0) * 100.0
        f1 = f1_score(all_labels, all_preds, zero_division=0) * 100.0
        
        try:
            roc_auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            roc_auc = 0.0
            
        cm = confusion_matrix(all_labels, all_preds)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = (tn / (tn + fp)) * 100.0 if (tn + fp) > 0 else 0.0
        else:
            specificity = 0.0

        result_line = f"Epoch {epoch_num:<1} | {accuracy:>6.2f}%   | {sensitivity:>9.2f}%   | {specificity:>9.2f}%   | {precision:>7.2f}%   | {f1:>7.2f}%   | {roc_auc:.4f}"
        print(result_line)
        
        with open(TEST_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(result_line + "\n")

    print("=" * 89)
    print(f"\n[HOÀN TẤT] Toàn bộ kết quả qua các Epoch đã được lưu tại: {TEST_LOG_FILE}")

if __name__ == "__main__":
    test_all_checkpoints()
