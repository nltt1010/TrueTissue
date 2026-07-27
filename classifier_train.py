import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

# Import mô hình phân loại và dataset từ các module của bạn
from classifier_model import ResNet18Classifier
from utils import ClassificationDataset

# --- CẤU HÌNH ĐƯỜNG DẪN & THAM SỐ ---
DATA_ROOT = Path(r"G:/CV/pj/dataset/train")
CHECKPOINT_DIR = Path(r"G:/CV/pj/cls_checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)

LOG_DIR = Path("./cls_log")
LOG_DIR.mkdir(exist_ok=True, parents=True)
LOG_FILE_PATH = LOG_DIR / "classifier_log.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOTAL_EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 0.0001

def train_classifier():
    print(f"[CẤU HÌNH] Đang sử dụng thiết bị: {device}")
    
    # 1. DataLoader
    train_dataset = ClassificationDataset(DATA_ROOT, img_size=256, mode='fake')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    num_classes = getattr(train_dataset, 'num_classes', 2)
    
    # 2. Khởi tạo mô hình & optimizer
    model = ResNet18Classifier(num_classes=num_classes, freeze_backbone=False).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # =====================================================================
    # --- CƠ CHẾ RESUME TRAINING TƯƠNG TỰ M_TRAIN.PY ---
    # =====================================================================
    start_epoch = 0
    best_acc = 0.0
    RESUME_CHECKPOINT = CHECKPOINT_DIR / "cls_last.pth"

    if RESUME_CHECKPOINT.exists():
        print(f"\n[RESUME] Tìm thấy file checkpoint cũ: {RESUME_CHECKPOINT.name}. Đang nạp dữ liệu...")
        checkpoint = torch.load(RESUME_CHECKPOINT, map_location=device)
        
        # Nạp trọng số mô hình và optimizer
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        # Nạp epoch và accuracy tốt nhất trước đó
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('best_acc', 0.0)
        
        print(f"[RESUME] Nạp thành công! Sẽ tiếp tục chạy từ Epoch {start_epoch} (Best Acc hiện tại: {best_acc:.2f}%)\n")
    else:
        print("\n[VỪA CHẠY] Không tìm thấy checkpoint cũ. Bắt đầu huấn luyện từ Epoch 0.\n")
    # =====================================================================

    print("BẮT ĐẦU HUẤN LUYỆN RESNET18 CLASSIFIER...")
    print("=" * 70)

    for epoch in range(start_epoch, TOTAL_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)

            # In progress theo batch nhỏ nếu muốn theo dõi
            if (i + 1) % 10 == 0 or (i + 1) == len(train_loader):
                batch_acc = (torch.sum(preds == labels.data).item() / labels.size(0)) * 100.0
                print(f"Epoch [{epoch}/{TOTAL_EPOCHS-1}] Iter [{i+1}/{len(train_loader)}] - Batch Loss: {loss.item():.4f} - Batch Acc: {batch_acc:.2f}%")

        # Tính chỉ số trung bình sau mỗi Epoch
        epoch_loss = running_loss / total
        epoch_acc = (correct / total) * 100.0

        # Ghi Log vào file txt
        log_message = f"Epoch {epoch} [{len(train_loader)}/{len(train_loader)}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%"
        print(f"\n---> KẾT QUẢ {log_message}\n")
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(log_message + "\n")

        # =====================================================================
        # --- CƠ CHẾ LƯU CHECKPOINT ĐÚNG CHUẨN ĐỒ ÁN ---
        # =====================================================================
        if epoch_acc > best_acc:
            best_acc = epoch_acc

        checkpoint_data = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_acc': best_acc,
            'num_classes': num_classes
        }

        # 1. Lưu file theo từng epoch để lưu lịch sử (cls_0.pth, cls_1.pth...)
        torch.save(checkpoint_data, CHECKPOINT_DIR / f"cls_{epoch}.pth")
        
        # 2. Lưu/Ghi đè file cls_last.pth để tự động Resume khi bị gián đoạn
        torch.save(checkpoint_data, CHECKPOINT_DIR / "cls_last.pth")
        
        # 3. Nếu là epoch có kết quả tốt nhất, lưu thêm bản cls_best.pth
        if epoch_acc == best_acc:
            torch.save(checkpoint_data, CHECKPOINT_DIR / "cls_best.pth")
            print(f"[SAVE BEST] Đã lưu mô hình có Accuracy cao nhất ({best_acc:.2f}%) tại: cls_best.pth")

        print(f"[SAVE] Đã lưu thành công trạng thái Epoch {epoch} tại: {CHECKPOINT_DIR}\n" + "-"*70)

if __name__ == "__main__":
    train_classifier()