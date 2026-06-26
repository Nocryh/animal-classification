"""
Export PyTorch model to ONNX format
=====================================
Converts a trained ResNet50 .pth checkpoint to ONNX for deployment.

Usage:
    python export_onnx.py
    python export_onnx.py --model_path ./my_model.pth --classes_file ./classes.txt
"""
import argparse
import os

import torch
import torch.nn as nn
from torchvision import models


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a ResNet50 animal classifier checkpoint to ONNX"
    )
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to trained .pth checkpoint (default: animal_classifier.pth in script dir)")
    parser.add_argument("--classes_file", type=str, default=None,
                        help="Path to class names text file (default: animal_classifier_classes.txt in script dir)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: script directory)")
    parser.add_argument("--num_classes", type=int, default=90)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--opset", type=int, default=14)
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = args.output_dir or script_dir

    model_path = args.model_path or os.path.join(script_dir, "animal_classifier.pth")
    classes_file = args.classes_file or os.path.join(script_dir, "animal_classifier_classes.txt")
    output_onnx = os.path.join(output_dir, "animal_classifier.onnx")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load class names
    with open(classes_file, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]

    if len(class_names) != args.num_classes:
        print(f"Warning: Expected {args.num_classes} classes, found {len(class_names)}")

    # Load model
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, args.num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    print(f"Model loaded from: {model_path}")

    # Export ONNX
    dummy_input = torch.randn(1, 3, args.img_size, args.img_size, device=device)
    torch.onnx.export(
        model, dummy_input, output_onnx,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=args.opset,
    )
    print(f"ONNX exported to: {output_onnx}")

    # Verify
    import onnx
    onnx_model = onnx.load(output_onnx)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verification passed.")


if __name__ == "__main__":
    main()
