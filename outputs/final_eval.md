# Final evaluation: mobilenetv2_224.keras

Test set: 1654 images (1299 without a look-alike in the original Train/Val). 95% bootstrap intervals, 1000 resamples.

| Evaluation set | Accuracy | Macro-F1 | Weakest-class recall |
|---|---|---|---|
| All of Test | 0.953 (0.943 to 0.963) | 0.954 (0.943 to 0.964) | 0.929 |
| Test without look-alikes | 0.950 (0.938 to 0.962) | 0.951 (0.939 to 0.963) | 0.920 |

## Per-class results (all of Test)

| Class | Precision | Recall | F1 | Images |
|---|---|---|---|---|
| Earthquake / collapse | 0.939 | 0.971 | 0.955 | 239 |
| Flood | 0.945 | 0.958 | 0.952 | 502 |
| Normal | 0.949 | 0.929 | 0.939 | 477 |
| Wildfire / fire | 0.977 | 0.966 | 0.971 | 436 |

## The costly mistake

- A real disaster (flood, fire or collapse) called **Normal**: 2.0% of 1177 disaster images.
- A normal scene flagged as a disaster: 7.1% of 477 normal images.

## Calibration

Mean confidence 0.953 against accuracy 0.953; expected calibration error 0.011.
3.2% of images get a confidence below 0.6 (the app flags these for a manual check).

## Urgency score (default weights)

Share of images with a score of at least 40, by true class: Earthquake / collapse 99%, Flood 99%, Normal 9%, Wildfire / fire 99%
