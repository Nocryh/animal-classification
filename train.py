"""
Animal Classification Training Script
======================================
Trains a ResNet50 model on a custom animal dataset, exports to ONNX format.

Usage:
    python train.py --data_dir ./data/animals --classes_file ./data/classes.txt
    python train.py --data_dir ./data/animals --classes_file ./data/classes.txt --epochs 30 --lr 1e-4
"""
import argparse
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a ResNet50 animal classifier with ONNX export"
    )
    # Data paths
    parser.add_argument("--data_dir", type=str, default="./data/animals",
                        help="Path to ImageFolder-style dataset directory")
    parser.add_argument("--classes_file", type=str, default="./data/classes.txt",
                        help="Path to text file with class names (one per line)")
    # Output
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save model and class names (default: script directory)")
    # Training hyperparams
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_classes", type=int, default=90)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    # Output directory
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    output_onnx = os.path.join(output_dir, "animal_classifier.onnx")
    output_model = os.path.join(output_dir, "animal_classifier.pth")
    class_names_path = os.path.join(output_dir, "animal_classifier_classes.txt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load class names
    with open(args.classes_file, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]

    # Validate class count
    dataset = datasets.ImageFolder(args.data_dir)
    if len(dataset.classes) != args.num_classes:
        print(f"Warning: Expected {args.num_classes} classes, found {len(dataset.classes)}")
    print(f"Dataset: {len(dataset)} images, {len(dataset.classes)} classes")
    print(f"Device: {device}")

    # Train/val split per class (stratified)
    train_indices, val_indices = [], []
    for cls_idx in range(len(dataset.classes)):
        cls_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == cls_idx]
        n_val = max(1, int(len(cls_indices) * args.val_split))
        perm = np.random.RandomState(args.seed).permutation(len(cls_indices))
        val_indices.extend([cls_indices[i] for i in perm[:n_val]])
        train_indices.extend([cls_indices[i] for i in perm[n_val:]])

    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = Subset(
        datasets.ImageFolder(args.data_dir, transform=train_transform), train_indices
    )
    val_dataset = Subset(
        datasets.ImageFolder(args.data_dir, transform=val_transform), val_indices
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}")

    # Model
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, args.num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        # Training
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        scheduler.step()
        train_loss = running_loss / len(train_indices)

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total * 100

        print(f"Epoch {epoch:2d}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}%",
              end="")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), output_model)
            print("  [Best!]", end="")
        print()

    print(f"\nTraining complete. Best val acc: {best_acc:.2f}%")

    # Export ONNX
    model.load_state_dict(torch.load(output_model, map_location=device, weights_only=True))
    model.eval()

    dummy_input = torch.randn(1, 3, args.img_size, args.img_size, device=device)
    torch.onnx.export(
        model, dummy_input, output_onnx,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=14,
    )
    print(f"ONNX model saved to: {output_onnx}")

    # Verify ONNX
    import onnx
    onnx_model = onnx.load(output_onnx)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verification passed.")

    # Save class names
    with open(class_names_path, "w", encoding="utf-8") as f:
        f.write("\n".join(class_names))
    print(f"Class names saved to: {class_names_path}")


if __name__ == "__main__":
    main()
