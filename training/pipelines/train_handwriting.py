import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ml.handwriting.dataset import HandwrittenMedicineDataset
from ml.handwriting.model import CRNNHandwritingClassifier
from ml.utils.provenance import compute_directory_sha256, get_hardware_info


def compute_top_k_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
    """Compute top-k accuracy."""
    with torch.no_grad():
        _, pred = logits.topk(min(k, logits.size(1)), dim=1, largest=True, sorted=True)
        correct = pred.eq(targets.view(-1, 1).expand_as(pred))
        return correct.any(dim=1).float().mean().item()


def train_handwriting_model(
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 42,
    device_name: str = "auto"
) -> Dict[str, Any]:
    """Execute complete training and independent test evaluation for handwritten medicine recognition."""
    torch.manual_seed(seed)
    start_time = time.time()
    
    root_dir = Path(__file__).resolve().parent.parent.parent
    data_dir = root_dir / "data" / "raw" / "bd_handwritten"
    checkpoints_dir = root_dir / "checkpoints" / "handwriting"
    weights_dir = root_dir / "ml" / "weights"
    reports_dir = root_dir / "reports" / "training"
    
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Hardware detection
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    print(f"[*] Training Model B (Handwritten Medicine Recognition) on device: {device}")

    # 1. Load Datasets
    train_dataset = HandwrittenMedicineDataset(base_dir=data_dir, split="Training")
    val_dataset = HandwrittenMedicineDataset(base_dir=data_dir, split="Validation", label_map=train_dataset.label_map)
    test_dataset = HandwrittenMedicineDataset(base_dir=data_dir, split="Testing", label_map=train_dataset.label_map)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_classes = len(train_dataset.label_map)
    print(f"[*] Total classes: {num_classes} | Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    # 2. Initialize Model
    model = CRNNHandwritingClassifier(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # 3. Training Loop
    history = []
    best_val_acc = 0.0
    best_checkpoint_path = checkpoints_dir / "best_handwriting_crnn.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, total_train = 0.0, 0, 0

        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            total_train += images.size(0)

        epoch_train_loss = train_loss / max(total_train, 1)
        epoch_train_acc = train_correct / max(total_train, 1)

        # Validation Step
        model.eval()
        val_loss, val_correct, total_val = 0.0, 0, 0
        top5_correct = 0.0

        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                top5_correct += compute_top_k_accuracy(logits, labels, k=5) * images.size(0)
                total_val += images.size(0)

        epoch_val_loss = val_loss / max(total_val, 1)
        epoch_val_acc = val_correct / max(total_val, 1)
        epoch_val_top5 = top5_correct / max(total_val, 1)

        history.append({
            "epoch": epoch,
            "train_loss": round(epoch_train_loss, 4),
            "train_acc": round(epoch_train_acc, 4),
            "val_loss": round(epoch_val_loss, 4),
            "val_top1_acc": round(epoch_val_acc, 4),
            "val_top5_acc": round(epoch_val_top5, 4)
        })

        print(f"Epoch {epoch}/{epochs} | Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f} Top-1: {epoch_val_acc:.4f} Top-5: {epoch_val_top5:.4f}")

        if epoch_val_acc >= best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "label_map": train_dataset.label_map,
                "num_classes": num_classes,
                "val_acc": epoch_val_acc
            }, best_checkpoint_path)

    # 4. Independent Test Split Evaluation (Held-out, untouched)
    print("\n[*] Evaluating Best Checkpoint on Hold-Out Test Split...")
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_loss, test_correct, total_test = 0.0, 0, 0
    test_top5 = 0.0

    with torch.no_grad():
        for images, labels, _ in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            test_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            test_correct += (preds == labels).sum().item()
            test_top5 += compute_top_k_accuracy(logits, labels, k=5) * images.size(0)
            total_test += images.size(0)

    final_test_top1 = test_correct / max(total_test, 1)
    final_test_top5 = test_top5 / max(total_test, 1)
    final_test_loss = test_loss / max(total_test, 1)

    # Also save portable model weights for inference
    torch.save(model.state_dict(), weights_dir / "crnn_handwriting.pth")
    with open(weights_dir / "medicine_vocab.json", "w", encoding="utf-8") as f:
        json.dump(train_dataset.label_map, f, indent=2)

    elapsed_time = round(time.time() - start_time, 2)
    dataset_hash = compute_directory_sha256(data_dir)
    hardware_info = get_hardware_info()

    report = {
        "pipeline_name": "Handwritten Medicine Recognition",
        "model_id": "Model_B_CRNN_BiLSTM",
        "dataset_name": "Doctor's Handwritten Prescription BD",
        "dataset_version": "v1.0",
        "dataset_sha256": dataset_hash,
        "training_seed": seed,
        "training_time_seconds": elapsed_time,
        "hardware_information": hardware_info,
        "model_configuration": {
            "architecture": "CRNN_CNN_BiLSTM",
            "num_classes": num_classes,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "optimizer": "AdamW",
            "loss_fn": "CrossEntropyLoss"
        },
        "training_metrics": {
            "final_train_loss": history[-1]["train_loss"] if history else None,
            "final_train_acc": history[-1]["train_acc"] if history else None,
            "epoch_history": history
        },
        "validation_metrics": {
            "best_val_top1_accuracy": round(best_val_acc, 4),
            "final_val_top5_accuracy": history[-1]["val_top5_acc"] if history else None,
            "final_val_loss": history[-1]["val_loss"] if history else None
        },
        "test_metrics": {
            "test_top1_accuracy": round(final_test_top1, 4),
            "test_top5_accuracy": round(final_test_top5, 4),
            "test_loss": round(final_test_loss, 4),
            "test_samples_evaluated": total_test
        },
        "checkpoint_path": str(best_checkpoint_path.relative_to(root_dir))
    }

    report_path = reports_dir / "handwriting_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[+] Training complete in {elapsed_time}s! Report saved to {report_path}")
    return report


if __name__ == "__main__":
    train_handwriting_model(epochs=3, batch_size=32)
