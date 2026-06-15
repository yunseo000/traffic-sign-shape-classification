from pathlib import Path
import sys

import numpy as np

from features import image_to_features
from train import predict


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "outputs" / "model.npz"


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <image_path>")
        print("Example: python src/predict.py data/sign_shapes/circle/circle_000.png")
        return

    image_path = Path(sys.argv[1])
    model = np.load(MODEL_PATH, allow_pickle=True)
    network = {
        "w1": model["w1"],
        "b1": model["b1"],
        "w2": model["w2"],
        "b2": model["b2"],
    }
    classes = model["classes"]

    x = image_to_features(image_path).reshape(1, -1)
    pred, probs = predict(x, network)
    label = classes[int(pred[0])]
    confidence = float(probs[0, int(pred[0])])
    print(f"Prediction: {label} ({confidence:.2%})")


if __name__ == "__main__":
    main()
