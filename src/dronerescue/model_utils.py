"""Load the trained model, classify photos, and explain predictions with Grad-CAM."""
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from . import config as C
from .priority import level, priority_score

MODEL_NAMES = ["mobilenetv2_224.keras", "mobilenetv2_128.keras"]


def find_model(explicit=None):
    """Return the path of the trained model, or None. Checks $DRONERESCUE_MODEL, then the usual output folders."""
    explicit = explicit or os.environ.get("DRONERESCUE_MODEL")
    if explicit:
        return Path(explicit) if Path(explicit).exists() else None
    folders = [C.OUTPUT_DIR, C.PROJECT_ROOT / "outputs", C.PROJECT_ROOT / "models", C.PROJECT_ROOT]
    for folder in folders:
        for name in MODEL_NAMES:
            if (folder / name).exists():
                return folder / name
    return None


def load_model(path):
    from tensorflow import keras
    return keras.models.load_model(path, compile=False)


def model_input_size(model):
    return int(model.input_shape[1])


def preprocess(pil_image, size):
    """Same preprocessing as the training cache: RGB, squashed to a size x size square (pixel values 0-255)."""
    return np.asarray(pil_image.convert("RGB").resize((size, size), Image.LANCZOS), dtype=np.float32)


def predict_probs(model, arrays, batch=32):
    arrays = np.asarray(arrays, dtype=np.float32)
    return np.concatenate([model.predict(arrays[i:i + batch], verbose=0) for i in range(0, len(arrays), batch)])


def demo_probs(image_bytes_list):
    """Placeholder used only when no trained model is available. NOT a prediction: random but repeatable per image."""
    out = []
    for b in image_bytes_list:
        rng = np.random.default_rng(int(hashlib.md5(b).hexdigest()[:8], 16))
        out.append(rng.dirichlet([0.6] * len(C.CLASSES)))
    return np.array(out)


def analyse(probs, names, weights):
    """Turn probabilities into a ranked table: class, confidence, priority score and level."""
    rows = []
    for name, p in zip(names, probs):
        s = priority_score(p, weights)
        lvl = level(s)[0]
        k = int(np.argmax(p))
        rows.append({"file": name, "predicted": C.DISPLAY_NAMES[C.CLASSES[k]], "confidence": round(float(p[k]), 3),
                     "priority": round(s, 1), "level": lvl,
                     **{f"p_{c}": round(float(p[i]), 3) for i, c in enumerate(C.CLASSES)}})
    df = pd.DataFrame(rows).sort_values("priority", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    return df


# ---------------------------------------------------------------- Grad-CAM
def _parts(model):
    from tensorflow import keras
    rescale = next(l for l in model.layers if isinstance(l, keras.layers.Rescaling))
    base = next(l for l in model.layers if l.name.startswith("mobilenetv2"))
    return rescale, base, model.layers[-1]


def gradcam(model, array, class_idx=None):
    """Heatmap (values 0-1, shape of the base's feature map) showing which regions pushed the model toward a class."""
    import tensorflow as tf
    rescale, base, dense = _parts(model)
    x = rescale(tf.convert_to_tensor(array[None], dtype=tf.float32))
    with tf.GradientTape() as tape:
        fmap = base(x, training=False)
        probs = dense(tf.reduce_mean(fmap, axis=(1, 2)))
        if class_idx is None:
            class_idx = int(tf.argmax(probs[0]))
        score = probs[:, class_idx]
    grads = tape.gradient(score, fmap)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.nn.relu(tf.reduce_sum(fmap * weights[:, None, None, :], axis=-1))[0].numpy()
    return cam / cam.max() if cam.max() > 0 else cam


def overlay(array, heatmap, alpha=0.45):
    """Blend a Grad-CAM heatmap over the photo (array uint8 or float 0-255, shape HxWx3)."""
    import matplotlib
    h, w = array.shape[:2]
    cam = np.asarray(Image.fromarray(heatmap.astype(np.float32), mode="F").resize((w, h), Image.BICUBIC))
    cam = np.clip(cam, 0, 1)
    colour = (matplotlib.colormaps["jet"](cam)[..., :3] * 255).astype(np.float32)
    return np.clip((1 - alpha) * array.astype(np.float32) + alpha * colour, 0, 255).astype(np.uint8)
