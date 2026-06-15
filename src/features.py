from pathlib import Path

import numpy as np
from PIL import Image


CLASSES = ["circle", "triangle", "square"]


def image_to_features(path, feature_size=32):
    img = Image.open(path).convert("L").resize((feature_size, feature_size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    # Invert so the sign shape has larger values than the bright background.
    arr = 1.0 - arr
    return arr.reshape(-1)


def load_dataset(data_dir):
    data_dir = Path(data_dir)
    features = []
    labels = []
    paths = []

    for label_idx, label in enumerate(CLASSES):
        for path in sorted((data_dir / label).glob("*.png")):
            features.append(image_to_features(path))
            labels.append(label_idx)
            paths.append(path)

    if not features:
        raise FileNotFoundError(f"No PNG images found under {data_dir}")

    return np.vstack(features), np.array(labels, dtype=np.int64), paths


def train_test_split(x, y, paths, test_ratio=0.25, seed=7):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(y))
    rng.shuffle(indices)
    test_size = int(len(indices) * test_ratio)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]

    return (
        x[train_idx],
        x[test_idx],
        y[train_idx],
        y[test_idx],
        [paths[i] for i in train_idx],
        [paths[i] for i in test_idx],
    )
