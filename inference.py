import onnxruntime as ort
import numpy as np
from PIL import Image
import sys
import os

# ======================== CONFIG ========================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ONNX_MODEL_PATH = os.path.join(PROJECT_DIR, "animal_classifier.onnx")
CLASS_NAMES_PATH = os.path.join(PROJECT_DIR, "animal_classifier_classes.txt")
IMG_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
TOP_K = 5
# =========================================================


class AnimalClassifier:
    def __init__(self, model_path=ONNX_MODEL_PATH, class_names_path=CLASS_NAMES_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        with open(class_names_path, "r", encoding="utf-8") as f:
            self.class_names = [line.strip() for line in f if line.strip()]

    def preprocess(self, image_path):
        img = Image.open(image_path).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        img_np = np.array(img, dtype=np.float32) / 255.0
        img_np = (img_np - MEAN) / STD
        img_np = img_np.transpose(2, 0, 1)  # HWC -> CHW
        img_np = np.expand_dims(img_np, axis=0)  # add batch dim
        return img_np.astype(np.float32)

    def predict(self, image_path):
        input_tensor = self.preprocess(image_path)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        logits = outputs[0][0]
        probs = self.softmax(logits)
        top_indices = np.argsort(probs)[::-1][:TOP_K]
        return [(self.class_names[i], probs[i]) for i in top_indices]

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        sys.exit(1)

    classifier = AnimalClassifier()
    results = classifier.predict(image_path)

    print(f"\nImage: {image_path}")
    print("-" * 40)
    for rank, (name, prob) in enumerate(results, 1):
        bar = "█" * int(prob * 40)
        print(f"  {rank:2d}. {name:<20s} {prob:.2%}  {bar}")
