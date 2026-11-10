"""Project-wide constants. Everything that other scripts need to agree on lives here."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Where the dataset lives. On Colab, point this at Google Drive so the 1.5 GB download
# survives a runtime reset:  export DRONERESCUE_DATA=/content/drive/MyDrive/dronerescue_data
DATA_DIR = Path(os.environ.get("DRONERESCUE_DATA", PROJECT_ROOT / "data"))
# Models, plots and result files. On Colab, point this at Drive too, or they vanish when the session resets:
#   export DRONERESCUE_OUTPUT=/content/drive/MyDrive/dronerescue_outputs
OUTPUT_DIR = Path(os.environ.get("DRONERESCUE_OUTPUT", PROJECT_ROOT / "outputs"))

# --- AIDERv2 (Zenodo record 10891054, licence CC BY 4.0) ---------------------
ZENODO_BASE = "https://zenodo.org/records/10891054/files"
DATASET_DIRNAME = "AIDERv2"
SPLITS = ["Train", "Val", "Test"]
CLASSES = ["Earthquake", "Flood", "Normal", "Wildfire"]    # folder names in the dataset
DISPLAY_NAMES = {"Earthquake": "Earthquake / collapse", "Flood": "Flood",
                 "Normal": "Normal", "Wildfire": "Wildfire / fire"}
# Folder names seen (or plausible) in the archives, mapped to our canonical class names.
# The Zenodo text says "Wildfire" but other descriptions of the same data say "Fire".
CLASS_ALIASES = {
    "earthquake": "Earthquake", "earthquakes": "Earthquake", "collapsed_building": "Earthquake",
    "collapsed_buildings": "Earthquake", "collapse": "Earthquake",
    "flood": "Flood", "floods": "Flood",
    "normal": "Normal", "non_disaster": "Normal",
    "wildfire": "Wildfire", "wildfires": "Wildfire", "fire": "Wildfire", "fires": "Wildfire",
}
CLASS_COLORS = {"Earthquake": "#f5b342", "Flood": "#4a8cff",
                "Normal": "#2ee6be", "Wildfire": "#ff6b4a"}

ZIP_FILES = {   # split -> (file name, md5 published on Zenodo)
    "Train": ("Train.zip", "25d0357a1d3c378fbbeb1a9c543d5a45"),
    "Val": ("Val.zip", "8fb4daca73b8ffb9c745a4dea721d08f"),
    "Test": ("Test.zip", "30d338e94dd0fbcc55ca67315981a900"),
}

# Image counts published on the Zenodo page, used as a sanity check after download
EXPECTED_COUNTS = {
    "Train": {"Earthquake": 1927, "Flood": 4063, "Wildfire": 3509, "Normal": 3900},
    "Val":   {"Earthquake": 239,  "Flood": 505,  "Wildfire": 439,  "Normal": 487},
    "Test":  {"Earthquake": 239,  "Flood": 502,  "Wildfire": 436,  "Normal": 477},
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# --- duplicate detection -----------------------------------------------------
HASH_MAX_DISTANCE = 5     # max differing bits (of 64) for two dHashes to be candidates
THUMB_MAX_MAD = 12.0      # candidates must also have a close 16x16 grey thumbnail (0..255 scale)
MIN_SIDE_PX = 32          # images with a shorter side than this are flagged as too small
RANDOM_SEED = 42

# --- modelling (Phase 1 onwards) ---------------------------------------------
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
DEFAULT_IMG_SIZE = 128    # images are squashed to a square of this size and cached as uint8 arrays
