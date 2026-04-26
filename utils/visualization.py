from pathlib import Path

import numpy as np
from PIL import Image


def save_mask(prob_map, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prob_map = np.asarray(prob_map)
    prob_map = np.clip(prob_map * 255.0, 0, 255).astype("uint8")
    Image.fromarray(prob_map).save(path)


def save_prediction_visualization(image, prob_map, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.asarray(image).astype("uint8")
    prob_map = np.asarray(prob_map)
    heat = np.zeros_like(image)
    heat[..., 0] = np.clip(prob_map * 255.0, 0, 255).astype("uint8")
    blended = (0.6 * image + 0.4 * heat).astype("uint8")
    Image.fromarray(blended).save(path)

