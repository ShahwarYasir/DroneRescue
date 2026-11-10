"""The urgency score. It is a hand-set rule applied to the model's class probabilities, NOT something learned:
no public dataset has severity labels. The weights are an assumption and the app lets the user change them."""
from . import config as C

DEFAULT_WEIGHTS = {"Earthquake": 1.0, "Flood": 0.9, "Wildfire": 0.9, "Normal": 0.0}

LEVELS = [   # (minimum score, name, colour, suggested action)
    (70, "High", "#ff6b4a", "Send a team to check this area first."),
    (40, "Medium", "#f5b342", "Review soon and look for follow-up photos."),
    (0, "Low", "#2ee6be", "No immediate action suggested."),
]


def priority_score(probs, weights=None):
    """0-100 urgency: probability-weighted sum of the class weights. `probs` follows the order of C.CLASSES."""
    w = weights or DEFAULT_WEIGHTS
    return float(100.0 * sum(p * w[c] for p, c in zip(probs, C.CLASSES)))


def level(score):
    for minimum, name, colour, advice in LEVELS:
        if score >= minimum:
            return name, colour, advice
    return LEVELS[-1][1:]
