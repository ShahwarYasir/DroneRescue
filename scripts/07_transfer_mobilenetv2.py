"""Phase 2: transfer learning with MobileNetV2 (ImageNet weights).

Stage 1 (feature extraction): the pretrained base is frozen and only a new classification head is trained.
Stage 2 (fine-tuning): the last --unfreeze layers of the base are trained too, at a very small learning rate.
BatchNorm layers stay frozen in stage 2, which is the usual recipe for stable fine-tuning.
The stage with the better VALIDATION macro-F1 is kept; Test is only used for the final report.

    python scripts/07_transfer_mobilenetv2.py --size 128 --tag mobilenetv2_128
    python scripts/07_transfer_mobilenetv2.py --size 224 --tag mobilenetv2_224    # needs: 03_build_cache.py --size 224

Needs a GPU. The ImageNet weights are downloaded automatically the first time.
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
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--epochs-head", type=int, default=8)
ap.add_argument("--epochs-ft", type=int, default=25)
ap.add_argument("--unfreeze", type=int, default=40, help="number of final base layers to fine-tune")
ap.add_argument("--lr-head", type=float, default=1e-3)
ap.add_argument("--lr-ft", type=float, default=3e-5)
ap.add_argument("--tag", default=None, help="name for outputs (default mobilenetv2_<size>)")
ap.add_argument("--variant", choices=["cleaned", "original"], default="cleaned")
ap.add_argument("--no-class-weights", action="store_true")
args = ap.parse_args()
tag = args.tag or f"mobilenetv2_{args.size}"
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

# ---------------------------------------------------------------- model
base = keras.applications.MobileNetV2(input_shape=(args.size, args.size, 3), include_top=False, weights="imagenet")
base.trainable = False
inputs = keras.Input(shape=(args.size, args.size, 3))
x = keras.Sequential([layers.RandomFlip("horizontal_and_vertical"), layers.RandomRotation(0.1),
                      layers.RandomZoom(0.1), layers.RandomContrast(0.15)], name="augment")(inputs)
x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)            # MobileNetV2 expects pixel values in -1..1
x = base(x, training=False)                                # keeps BatchNorm in inference mode
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(C.CLASSES), activation="softmax")(x)
model = keras.Model(inputs, outputs, name=tag)


def n_trainable():
    return int(sum(np.prod(w.shape) for w in model.trainable_weights))


def score():
    return evaluate_all(y_val, model.predict(val_ds, verbose=0).argmax(1),
                        y_te, model.predict(test_ds, verbose=0).argmax(1), exposed)


def fit(epochs, lr, patience, reduce_lr):
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    cbs = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)]
    if reduce_lr:
        cbs.append(keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6))
    t = time.time()
    h = model.fit(train_ds, validation_data=val_ds, epochs=epochs, class_weight=cw, callbacks=cbs, verbose=2)
    return h.history, time.time() - t


# ---------------------------------------------------------------- stage 1: train the head only
print(f"\n=== Stage 1: frozen base, training the head ({n_trainable():,} trainable parameters)")
head_params = n_trainable()
h1, t1 = fit(args.epochs_head, args.lr_head, patience=3, reduce_lr=False)
res_head = score()
head_weights = model.get_weights()
print(f"Stage 1  val macro-F1 {res_head['val']['macro_f1']:.4f}  test macro-F1 {res_head['test_all']['macro_f1']:.4f}")

# ---------------------------------------------------------------- stage 2: fine-tune the top of the base
base.trainable = True
for layer in base.layers[:-args.unfreeze]:
    layer.trainable = False
for layer in base.layers[-args.unfreeze:]:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False
print(f"\n=== Stage 2: fine-tuning the last {args.unfreeze} base layers ({n_trainable():,} trainable parameters)")
ft_params = n_trainable()
h2, t2 = fit(args.epochs_ft, args.lr_ft, patience=5, reduce_lr=True)
res_ft = score()
print(f"Stage 2  val macro-F1 {res_ft['val']['macro_f1']:.4f}  test macro-F1 {res_ft['test_all']['macro_f1']:.4f}")

# choose the stage on VALIDATION only
if res_ft["val"]["macro_f1"] >= res_head["val"]["macro_f1"]:
    selected, res = "finetune", res_ft
else:
    selected, res = "head", res_head
    model.set_weights(head_weights)
print(f"\nSelected stage (by validation macro-F1): {selected}")

res = dict(res)
res.update(
    selected_stage=selected, img_size=args.size, variant=args.variant, params=int(model.count_params()),
    trainable_params={"head": head_params, "finetune": ft_params}, class_weights_used=cw is not None,
    train_seconds=round(t1 + t2, 1), epochs_run={"head": len(h1["loss"]), "finetune": len(h2["loss"])},
    stages={"head": {k: res_head[k]["macro_f1"] for k in ("val", "test_all")},
            "finetune": {k: res_ft[k]["macro_f1"] for k in ("val", "test_all")}},
    settings={"unfreeze": args.unfreeze, "lr_head": args.lr_head, "lr_ft": args.lr_ft, "batch": args.batch},
)
print(f"Val   accuracy {res['val']['accuracy']:.4f}  macro-F1 {res['val']['macro_f1']:.4f}")
print(f"Test  accuracy {res['test_all']['accuracy']:.4f}  macro-F1 {res['test_all']['macro_f1']:.4f}  "
      f"min-class recall {res['test_all']['min_class_recall']:.4f}")
if "test_unexposed" in res:
    u = res["test_unexposed"]
    print(f"Test without look-alikes ({u['n']} images)  accuracy {u['accuracy']:.4f}  macro-F1 {u['macro_f1']:.4f}")
for c in C.CLASSES:
    print(f"  {c:<11} recall (Test) {res['test_all']['per_class'][c]['recall']:.3f}")

(out_dir / f"phase2_{tag}_results.json").write_text(json.dumps(res, indent=2))
model.save(out_dir / f"{tag}.keras")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
n1 = len(h1["loss"])
for ax, key, title in ((axes[0], "loss", "Loss"), (axes[1], "accuracy", "Accuracy")):
    ax.plot(h1[key] + h2[key], label="train")
    ax.plot(h1["val_" + key] + h2["val_" + key], label="validation")
    ax.axvline(n1 - 0.5, color="gray", ls="--", lw=1)
    ax.text(n1 - 0.4, ax.get_ylim()[0], " fine-tuning starts", fontsize=8, color="gray", va="bottom")
    ax.set_title(title); ax.set_xlabel("epoch"); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(out_dir / f"phase2_{tag}_curves.png", dpi=150)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
plot_confusion(axes[0], res["test_all"]["confusion_matrix"], f"{tag}: Test (all)")
if "test_unexposed" in res:
    plot_confusion(axes[1], res["test_unexposed"]["confusion_matrix"], f"{tag}: Test without look-alikes")
plt.tight_layout()
plt.savefig(out_dir / f"phase2_{tag}_confusion.png", dpi=150)
print(f"Saved model, results and plots to {out_dir}")
