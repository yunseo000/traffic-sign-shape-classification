from pathlib import Path
import math
import random

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "sign_shapes"
CLASSES = ["circle", "triangle", "square"]
IMAGE_SIZE = 64
IMAGES_PER_CLASS = 160
SEED = 42


def jitter_color(base):
    return tuple(max(0, min(255, channel + random.randint(-25, 25))) for channel in base)


def rotate_points(points, angle_deg, center):
    angle = math.radians(angle_deg)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    cx, cy = center
    rotated = []
    for x, y in points:
        dx = x - cx
        dy = y - cy
        rotated.append((cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a))
    return rotated


def make_background():
    base = np.random.normal(loc=225, scale=12, size=(IMAGE_SIZE, IMAGE_SIZE, 3))
    base = np.clip(base, 170, 255).astype(np.uint8)
    return Image.fromarray(base, mode="RGB")


def draw_shape(label):
    img = make_background()
    draw = ImageDraw.Draw(img)

    cx = IMAGE_SIZE // 2 + random.randint(-7, 7)
    cy = IMAGE_SIZE // 2 + random.randint(-7, 7)
    radius = random.randint(17, 23)
    fill = jitter_color((215, 45, 50))
    outline = jitter_color((245, 245, 245))
    width = random.randint(3, 5)

    if label == "circle":
        box = [cx - radius, cy - radius, cx + radius, cy + radius]
        draw.ellipse(box, fill=fill)
        draw.ellipse(box, outline=outline, width=width)
    elif label == "square":
        half = radius
        points = [
            (cx - half, cy - half),
            (cx + half, cy - half),
            (cx + half, cy + half),
            (cx - half, cy + half),
        ]
        points = rotate_points(points, random.uniform(-18, 18), (cx, cy))
        draw.polygon(points, fill=fill)
        draw.line(points + [points[0]], fill=outline, width=width)
    elif label == "triangle":
        height = radius * 2
        points = [
            (cx, cy - height * 0.62),
            (cx - radius * 1.15, cy + height * 0.48),
            (cx + radius * 1.15, cy + height * 0.48),
        ]
        points = rotate_points(points, random.uniform(-16, 16), (cx, cy))
        draw.polygon(points, fill=fill)
        draw.line(points + [points[0]], fill=outline, width=width)
    else:
        raise ValueError(f"Unknown label: {label}")

    if random.random() < 0.35:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.7)))

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.85, 1.15))
    return img


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    for label in CLASSES:
        class_dir = DATA_DIR / label
        class_dir.mkdir(parents=True, exist_ok=True)
        for old_file in class_dir.glob("*.png"):
            old_file.unlink()

        for idx in range(IMAGES_PER_CLASS):
            img = draw_shape(label)
            img.save(class_dir / f"{label}_{idx:03d}.png")

    total = IMAGES_PER_CLASS * len(CLASSES)
    print(f"Generated {total} images in {DATA_DIR}")


if __name__ == "__main__":
    main()
