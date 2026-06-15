from pathlib import Path
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from features import CLASSES, load_dataset, train_test_split


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "sign_shapes"
OUTPUT_DIR = ROOT / "outputs"
MODEL_PATH = OUTPUT_DIR / "model.npz"


def softmax(logits):
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def one_hot(y, num_classes):
    out = np.zeros((len(y), num_classes), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out


def train_neural_network(x_train, y_train, epochs=900, lr=0.18, l2=0.0005, hidden_size=96):
    rng = np.random.default_rng(12)
    num_features = x_train.shape[1]
    num_classes = len(CLASSES)
    w1 = rng.normal(0, 0.04, size=(num_features, hidden_size))
    b1 = np.zeros(hidden_size)
    w2 = rng.normal(0, 0.04, size=(hidden_size, num_classes))
    b2 = np.zeros(num_classes)
    y_one_hot = one_hot(y_train, num_classes)
    losses = []

    for epoch in range(epochs):
        hidden_raw = x_train @ w1 + b1
        hidden = np.maximum(hidden_raw, 0)
        logits = hidden @ w2 + b2
        probs = softmax(logits)
        loss = -np.mean(np.sum(y_one_hot * np.log(probs + 1e-9), axis=1))
        loss += l2 * (np.sum(w1 * w1) + np.sum(w2 * w2))
        losses.append(float(loss))

        grad_logits = (probs - y_one_hot) / len(x_train)
        grad_w2 = hidden.T @ grad_logits + 2 * l2 * w2
        grad_b2 = grad_logits.sum(axis=0)
        grad_hidden = grad_logits @ w2.T
        grad_hidden[hidden_raw <= 0] = 0
        grad_w1 = x_train.T @ grad_hidden + 2 * l2 * w1
        grad_b1 = grad_hidden.sum(axis=0)

        w1 -= lr * grad_w1
        b1 -= lr * grad_b1
        w2 -= lr * grad_w2
        b2 -= lr * grad_b2

    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}, losses


def predict(x, model):
    hidden = np.maximum(x @ model["w1"] + model["b1"], 0)
    probs = softmax(hidden @ model["w2"] + model["b2"])
    return probs.argmax(axis=1), probs


def confusion_matrix(y_true, y_pred):
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for true, pred in zip(y_true, y_pred):
        matrix[true, pred] += 1
    return matrix


def save_confusion_matrix_image(matrix, path):
    cell = 90
    margin = 130
    size = margin + cell * len(CLASSES) + 30
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    max_value = max(1, int(matrix.max()))

    draw.text((margin, 20), "Predicted label", fill=(30, 30, 30), font=font)
    draw.text((15, margin - 45), "True label", fill=(30, 30, 30), font=font)

    for i, label in enumerate(CLASSES):
        draw.text((margin + i * cell + 18, margin - 28), label, fill=(30, 30, 30), font=font)
        draw.text((28, margin + i * cell + 36), label, fill=(30, 30, 30), font=font)

    for row in range(len(CLASSES)):
        for col in range(len(CLASSES)):
            value = int(matrix[row, col])
            intensity = int(255 - 165 * (value / max_value))
            color = (intensity, intensity + 10, 255)
            x0 = margin + col * cell
            y0 = margin + row * cell
            x1 = x0 + cell
            y1 = y0 + cell
            draw.rectangle([x0, y0, x1, y1], fill=color, outline=(80, 80, 80))
            draw.text((x0 + 38, y0 + 37), str(value), fill=(0, 0, 0), font=font)

    img.save(path)


def save_sample_predictions(paths, y_true, y_pred, probs, path, count=12):
    thumbs = []
    for idx in range(min(count, len(paths))):
        img = Image.open(paths[idx]).convert("RGB").resize((96, 96))
        canvas = Image.new("RGB", (160, 132), "white")
        canvas.paste(img, (32, 0))
        draw = ImageDraw.Draw(canvas)
        true_label = CLASSES[int(y_true[idx])]
        pred_label = CLASSES[int(y_pred[idx])]
        conf = probs[idx, int(y_pred[idx])]
        color = (20, 120, 60) if true_label == pred_label else (180, 40, 40)
        draw.text((8, 100), f"T: {true_label}", fill=(30, 30, 30))
        draw.text((8, 116), f"P: {pred_label} {conf:.2f}", fill=color)
        thumbs.append(canvas)

    cols = 4
    rows = int(np.ceil(len(thumbs) / cols))
    grid = Image.new("RGB", (cols * 160, rows * 132), "white")
    for idx, thumb in enumerate(thumbs):
        grid.paste(thumb, ((idx % cols) * 160, (idx // cols) * 132))
    grid.save(path)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    x, y, paths = load_dataset(DATA_DIR)
    x_train, x_test, y_train, y_test, train_paths, test_paths = train_test_split(x, y, paths)

    model = train_neural_network(x_train, y_train)
    network, losses = model
    train_pred, _ = predict(x_train, network)
    test_pred, test_probs = predict(x_test, network)
    matrix = confusion_matrix(y_test, test_pred)

    train_acc = float(np.mean(train_pred == y_train))
    test_acc = float(np.mean(test_pred == y_test))

    class_accuracy = {}
    for idx, label in enumerate(CLASSES):
        mask = y_test == idx
        class_accuracy[label] = float(np.mean(test_pred[mask] == y_test[mask]))

    np.savez(MODEL_PATH, **network, classes=np.array(CLASSES))
    metrics = {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "class_accuracy": class_accuracy,
        "confusion_matrix": matrix.tolist(),
        "train_images": int(len(y_train)),
        "test_images": int(len(y_test)),
        "final_loss": losses[-1],
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_confusion_matrix_image(matrix, OUTPUT_DIR / "confusion_matrix.png")
    save_sample_predictions(test_paths, y_test, test_pred, test_probs, OUTPUT_DIR / "sample_predictions.png")

    print(f"Train accuracy: {train_acc:.3f}")
    print(f"Test accuracy: {test_acc:.3f}")
    print(f"Saved outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
