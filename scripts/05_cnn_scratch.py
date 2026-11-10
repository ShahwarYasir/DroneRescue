"""Phase 1, step 3: a small CNN trained from scratch.

Augmentation (flips in both directions, rotation, zoom, contrast) lives inside the model, so it is only active
during training. Class weights counter the imbalance, early stopping and learning-rate reduction watch the
validation loss, and the best weights are restored. Test is only used for the final report.

    python scripts/05_cnn_scratch.py --size 128            # use a GPU (Colab: Runtime > T4 GPU)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dronerescue import config as C
from dronerescue.metrics import evaluate_all, plot_confusion
from dronerescue.pipeline import class_weights, load_split

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default=str(C.DATA_DIR))
ap.add_argument("--out-dir", default=str(C.OUTPUT_DIR))
ap.add_argument("--size", type=int, default=C.DEFAULT_IMG_SIZE)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--no-class-weights", action="store_true", help="ablation: train without class weights")
ap.add_argument("--tag", default="cnn_scratch")
ap.add_argument("--variant", choices=["cleaned", "original"], default="cleaned",
                help="original: train on the uncleaned official split (leakage experiment)")
args = ap.parse_args()
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
keras.utils.set_random_seed(C.RANDOM_SEED)
print("GPU:", tf.config.list_physical_devices("GPU") or "none (this will be slow)")

x_tr, y_tr, _, _ = load_split(args.data_dir, args.size, "Train", args.variant)
x_val, y_val, _, _ = load_split(args.data_dir, args.size, "Val", args.variant)
x_te, y_te, exposed, _ = load_split(args.data_dir, args.size, "Test", args.variant)
cw = None if args.no_class_weights else class_weights(y_tr)
print("class weights:", cw)


def make_ds(x, y, shuffle=False):
    """Stream images from the in-memory arrays (a generator avoids TensorFlow's 2 GB limit on constant tensors)."""
    rng = np.random.default_rng(C.RANDOM_SEED)

    def gen():
        for i in (rng.permutation(len(x)) if shuffle else range(len(x))):
            yield x[i], y[i]
    sig = (tf.TensorSpec((args.size, args.size, 3), tf.uint8), tf.TensorSpec((), tf.int64))
    return tf.data.Dataset.from_generator(gen, output_signature=sig).batch(args.batch).prefetch(tf.data.AUTOTUNE)


train_ds, val_ds, test_ds = make_ds(x_tr, y_tr, True), make_ds(x_val, y_val), make_ds(x_te, y_te)


def conv_block(x, filters):
    for _ in range(2):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)
    return layers.Dropout(0.1)(x)


inputs = keras.Input(shape=(args.size, args.size, 3))
x = layers.Rescaling(1.0 / 255)(inputs)
x = keras.Sequential([layers.RandomFlip("horizontal_and_vertical"), layers.RandomRotation(0.1),
                      layers.RandomZoom(0.1), layers.RandomContrast(0.15)], name="augment")(x)
for f in (32, 64, 128, 192):
    x = conv_block(x, f)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(len(C.CLASSES), activation="softmax")(x)
model = keras.Model(inputs, outputs, name=args.tag)
model.summary()
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
]
t0 = time.time()
hist = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, class_weight=cw, callbacks=callbacks, verbose=2)
train_seconds = time.time() - t0
epochs_run = len(hist.history["loss"])

res = evaluate_all(y_val, model.predict(val_ds, verbose=0).argmax(1), y_te, model.predict(test_ds, verbose=0).argmax(1), exposed)
res.update(train_seconds=round(train_seconds, 1), epochs_run=epochs_run, params=int(model.count_params()),
           best_epoch=int(np.argmin(hist.history["val_loss"])) + 1, img_size=args.size,
           class_weights_used=cw is not None, variant=args.variant)
print(f"\nVal   accuracy {res['val']['accuracy']:.4f}  macro-F1 {res['val']['macro_f1']:.4f}")
print(f"Test  accuracy {res['test_all']['accuracy']:.4f}  macro-F1 {res['test_all']['macro_f1']:.4f}  "
      f"min-class recall {res['test_all']['min_class_recall']:.4f}")
if "test_unexposed" in res:
    u = res["test_unexposed"]
    print(f"Test without look-alikes ({u['n']} images)  accuracy {u['accuracy']:.4f}  macro-F1 {u['macro_f1']:.4f}")
for c in C.CLASSES:
    print(f"  {c:<11} recall (Test) {res['test_all']['per_class'][c]['recall']:.3f}")

(out_dir / f"phase1_{args.tag}_results.json").write_text(json.dumps(res, indent=2))
model.save(out_dir / f"{args.tag}_{args.size}.keras")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(hist.history["loss"], label="train"); axes[0].plot(hist.history["val_loss"], label="validation")
axes[0].axvline(res["best_epoch"] - 1, color="gray", ls="--", lw=1)
axes[0].set_title("Loss"); axes[0].set_xlabel("epoch")
axes[1].plot(hist.history["accuracy"], label="train"); axes[1].plot(hist.history["val_accuracy"], label="validation")
axes[1].set_title("Accuracy"); axes[1].set_xlabel("epoch")
for ax in axes:
    ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(out_dir / f"phase1_{args.tag}_curves.png", dpi=150)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
plot_confusion(axes[0], res["test_all"]["confusion_matrix"], f"{args.tag}: Test (all)")
if "test_unexposed" in res:
    plot_confusion(axes[1], res["test_unexposed"]["confusion_matrix"], f"{args.tag}: Test without look-alikes")
plt.tight_layout()
plt.savefig(out_dir / f"phase1_{args.tag}_confusion.png", dpi=150)
print(f"Saved model, results and plots to {out_dir}")
