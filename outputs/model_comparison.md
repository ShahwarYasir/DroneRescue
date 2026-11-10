# Model comparison

Models are chosen on the validation set only. 'Unexposed' means Test images without a look-alike in Train/Val.

| Model | Val macro-F1 | Test accuracy | Test macro-F1 | Test min-class recall | Unexposed macro-F1 | Gap (all - unexposed) |
|---|---|---|---|---|---|---|
| Logistic regression (features) | 0.752 | 0.755 | 0.751 | 0.665 | 0.734 | +0.016 |
| Random forest (features) | 0.842 | 0.854 | 0.850 | 0.808 | 0.821 | +0.029 |
| cnn_noweights | 0.891 | 0.914 | 0.908 | 0.874 | 0.898 | +0.010 |
| cnn_original | 0.892 | 0.893 | 0.885 | 0.845 | 0.868 | +0.018 |
| cnn_scratch | 0.889 | 0.901 | 0.895 | 0.868 | 0.879 | +0.016 |
| mobilenetv2_128 | 0.951 | 0.964 | 0.961 | 0.958 | 0.955 | +0.006 |
| mobilenetv2_224 | 0.945 | 0.953 | 0.954 | 0.929 | 0.951 | +0.003 |

## Per-class recall on Test (all / unexposed)

| Model | Earthquake | Flood | Normal | Wildfire |
|---|---|---|---|---|
| Logistic regression (features) | 0.83 / 0.83 | 0.76 / 0.76 | 0.66 / 0.65 | 0.80 / 0.75 |
| Random forest (features) | 0.81 / 0.80 | 0.87 / 0.87 | 0.87 / 0.85 | 0.84 / 0.71 |
| cnn_noweights | 0.87 / 0.87 | 0.94 / 0.94 | 0.92 / 0.91 | 0.89 / 0.84 |
| cnn_original | 0.92 / 0.92 | 0.84 / 0.85 | 0.90 / 0.88 | 0.93 / 0.85 |
| cnn_scratch | 0.92 / 0.92 | 0.90 / 0.90 | 0.87 / 0.85 | 0.92 / 0.85 |
| mobilenetv2_128 | 0.96 / 0.96 | 0.96 / 0.96 | 0.96 / 0.95 | 0.97 / 0.95 |
| mobilenetv2_224 | 0.97 / 0.97 | 0.96 / 0.96 | 0.93 / 0.92 | 0.97 / 0.96 |

Recall matters most here: a missed flood or collapse costs more than a false alarm.
