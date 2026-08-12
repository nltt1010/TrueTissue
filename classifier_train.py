import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Import mô hình phân loại và dataset từ các module của bạn
from classifier_model import ResNet18Classifier
from utils import ClassificationDataset

ROOT_DIR = Path(__file__).resolve().parent
DATA_ROOT = ROOT_DIR / "dataset" / "train"
CHECKPOINT_DIR = ROOT_DIR / "cls_checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)

LOG_DIR = ROOT_DIR / "cls_log"
LOG_DIR.mkdir(exist_ok=True, parents=True)
LOG_FILE_PATH = LOG_DIR / "classifier_log.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOTAL_EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 0.0001

def train_classifier():
    print(f"[CẤU HÌNH] Đang sử dụng thiết bị: {device}")
    
    train_dataset = ClassificationDataset(DATA_ROOT, img_size=256, mode='fake')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    num_classes = getattr(train_dataset, 'num_classes', 2)
    
    model = ResNet18Classifier(num_classes=num_classes, freeze_backbone=False).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Resume training
    start_epoch = 0
    best_acc = 0.0
    RESUME_CHECKPOINT = CHECKPOINT_DIR / "cls_last.pth"

    if RESUME_CHECKPOINT.exists():
        print(f"\n[RESUME] Đang load checkpoint: {RESUME_CHECKPOINT.name}...")
        checkpoint = torch.load(RESUME_CHECKPOINT, map_location=device)
        
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('best_acc', 0.0)
        
        print(f"[RESUME] Train tiếp từ epoch {start_epoch}\n")
    else:
        print("\n[START] Bắt đầu train từ Epoch 0.\n")

    print("BẮT ĐẦU HUẤN LUYỆN RESNET18 CLASSIFIER...")
    print("=" * 70)

    for epoch in range(start_epoch, TOTAL_EPOCHS):
        model.train()
        running_loss = 0.0
        
        all_labels = []
        all_preds = []
        all_probs = []

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, preds = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())

            if (i + 1) % 10 == 0 or (i + 1) == len(train_loader):
                batch_acc = (torch.sum(preds == labels.data).item() / labels.size(0)) * 100.0
                print(f"Epoch [{epoch}/{TOTAL_EPOCHS-1}] Iter [{i+1}/{len(train_loader)}] - Batch Loss: {loss.item():.4f} - Batch Acc: {batch_acc:.2f}%")

        epoch_loss = running_loss / len(all_labels)
        
        epoch_acc = accuracy_score(all_labels, all_preds) * 100.0
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

        log_message = (f"Epoch {epoch} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}% | "
                       f"Sens(Recall): {sensitivity:.2f}% | Spec: {specificity:.2f}% | "
                       f"Prec: {precision:.2f}% | F1: {f1:.2f}% | AUC: {roc_auc:.4f}")
        print(f"\n---> KẾT QUẢ {log_message}\n")
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(log_message + "\n")

        # Lưu checkpoint
        if epoch_acc > best_acc:
            best_acc = epoch_acc

        checkpoint_data = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_acc': best_acc,
            'num_classes': num_classes
        }

        torch.save(checkpoint_data, CHECKPOINT_DIR / f"cls_{epoch}.pth")
        torch.save(checkpoint_data, CHECKPOINT_DIR / "cls_last.pth")
        
        if epoch_acc == best_acc:
            torch.save(checkpoint_data, CHECKPOINT_DIR / "cls_best.pth")
            print(f"[SAVE BEST] Đã lưu mô hình có Accuracy cao nhất ({best_acc:.2f}%) tại: cls_best.pth")

        print(f"[SAVE] Đã lưu thành công trạng thái Epoch {epoch} tại: {CHECKPOINT_DIR}\n" + "-"*70)

if __name__ == "__main__":
    train_classifier()