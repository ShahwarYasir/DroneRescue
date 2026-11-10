"""DroneRescue: upload drone photos taken after a disaster, get a label and an urgency score for each.

Run from the project folder:   streamlit run app/app.py
The trained model is looked for in outputs/ (mobilenetv2_224.keras or mobilenetv2_128.keras), or set
DRONERESCUE_MODEL=/path/to/model.keras. Without a model the app runs in clearly marked DEMO MODE.
"""
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dronerescue import config as C
from dronerescue.model_utils import (analyse, demo_probs, find_model, gradcam, load_model, model_input_size,
                                     overlay, predict_probs, preprocess)
from dronerescue.priority import DEFAULT_WEIGHTS, LEVELS, level, priority_score

st.set_page_config(page_title="Drone Rescue", page_icon="🛰️", layout="wide")
OUT = C.OUTPUT_DIR
MAX_SURVEY = 200

st.markdown("""<style>
.block-container {padding-top: 2rem;}
h1 span.accent {color: #2ee6be;}
.banner {border:1px solid #f5b342; background:#2a2110; border-radius:10px; padding:10px 14px; color:#f5d88a;}
.card {border:1px solid #263044; background:#111827; border-radius:12px; padding:14px 18px;}
.big {font-size:2.2rem; font-weight:700; line-height:1.1;}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------- model
@st.cache_resource(show_spinner="Loading the model ...")
def get_model():
    path = find_model()
    if path is None:
        return None, None, "No trained model file found."
    try:
        m = load_model(path)
        return m, path, None
    except Exception as e:                       # noqa: BLE001 - show any loading problem in the UI
        return None, path, f"Could not load {path.name}: {e}"


model, model_path, model_error = get_model()
DEMO = model is None

# ---------------------------------------------------------------- sidebar
st.sidebar.markdown("## 🛰️ DRONE **RESCUE**")
page = st.sidebar.radio("Page", ["Analyse photo", "Survey mode", "Model insights", "About and limits"], label_visibility="collapsed")
st.sidebar.markdown("---")
if DEMO:
    st.sidebar.warning("DEMO MODE: no trained model loaded. Numbers are random placeholders.")
    if model_error:
        st.sidebar.caption(model_error)
else:
    st.sidebar.success(f"Model: {model_path.name} ({model_input_size(model)}px input)")

st.sidebar.markdown("**Urgency weights**")
st.sidebar.caption("The score is a rule, not a learned value. Change the weights to match your priorities.")
weights = {"Earthquake": st.sidebar.slider("Earthquake / collapse", 0.0, 1.0, DEFAULT_WEIGHTS["Earthquake"], 0.05),
           "Flood": st.sidebar.slider("Flood", 0.0, 1.0, DEFAULT_WEIGHTS["Flood"], 0.05),
           "Wildfire": st.sidebar.slider("Wildfire / fire", 0.0, 1.0, DEFAULT_WEIGHTS["Wildfire"], 0.05),
           "Normal": 0.0}
st.sidebar.markdown("---")
st.sidebar.caption("Decision support only. A person decides where help goes.")


def classify(images, names, raw_bytes):
    """Return class probabilities for a list of PIL images."""
    if DEMO:
        return demo_probs(raw_bytes)
    size = model_input_size(model)
    return predict_probs(model, np.stack([preprocess(im, size) for im in images]))


def prob_chart(p):
    fig, ax = plt.subplots(figsize=(5, 2.4))
    fig.patch.set_facecolor("#111827"); ax.set_facecolor("#111827")
    labels = [C.DISPLAY_NAMES[c] for c in C.CLASSES]
    ax.barh(labels[::-1], (np.array(p) * 100)[::-1], color=[C.CLASS_COLORS[c] for c in C.CLASSES][::-1])
    for i, v in enumerate((np.array(p) * 100)[::-1]):
        ax.text(min(v + 1, 88), i, f"{v:.0f}%", va="center", color="white", fontsize=9)
    ax.set_xlim(0, 100); ax.tick_params(colors="#c8d0e0"); ax.set_xlabel("probability (%)", color="#c8d0e0")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#3a4560")
    plt.tight_layout()
    return fig


st.markdown("# DRONE <span class='accent'>RESCUE</span>", unsafe_allow_html=True)
st.caption("Aerial disaster recognition for emergency response: what happened here, and how urgent is it?")
if DEMO:
    st.markdown("<div class='banner'><b>DEMO MODE.</b> No trained model was found, so every prediction below is a random "
                "placeholder. Put <code>mobilenetv2_224.keras</code> in <code>outputs/</code> (or set "
                "<code>DRONERESCUE_MODEL</code>) and restart.</div>", unsafe_allow_html=True)
    st.write("")

# ================================================================ analyse photo
if page == "Analyse photo":
    up = st.file_uploader("Upload an aerial photo", type=["jpg", "jpeg", "png"])
    if up is None:
        st.info("Upload a drone or satellite photo to see what the model thinks and how urgent it is.")
    else:
        raw = up.getvalue()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        probs = classify([img], [up.name], [raw])[0]
        k = int(np.argmax(probs))
        score = priority_score(probs, weights)
        lvl, colour, advice = level(score)
        left, right = st.columns([1, 1])
        with left:
            show_cam = st.toggle("Show where the model looked (Grad-CAM)", value=False, disabled=DEMO)
            if show_cam and not DEMO:
                arr = preprocess(img, model_input_size(model))
                st.image(overlay(arr.astype(np.uint8), gradcam(model, arr, k)), width="stretch",
                         caption="Warm colours mark regions that pushed the model toward its prediction")
            else:
                st.image(img, width="stretch", caption=up.name)
        with right:
            st.markdown(f"<div class='card'><div style='color:#9aa6bf'>Prediction</div>"
                        f"<div class='big' style='color:{C.CLASS_COLORS[C.CLASSES[k]]}'>{C.DISPLAY_NAMES[C.CLASSES[k]]}</div>"
                        f"<div style='color:#9aa6bf'>confidence {probs[k] * 100:.0f}%</div></div>", unsafe_allow_html=True)
            st.write("")
            st.markdown(f"<div class='card'><div style='color:#9aa6bf'>Urgency</div>"
                        f"<div class='big' style='color:{colour}'>{score:.0f} / 100 &nbsp; {lvl}</div>"
                        f"<div>{advice}</div></div>", unsafe_allow_html=True)
            st.progress(min(int(score), 100))
            if probs[k] < 0.6:
                st.warning("Low confidence: the model is unsure. Check this photo by eye.")
            st.pyplot(prob_chart(probs))
        with st.expander("How the urgency score is calculated"):
            st.write("Score = 100 x (sum over classes of probability x weight). With the current weights: " +
                     ", ".join(f"{C.DISPLAY_NAMES[c]} {weights[c]:.2f}" for c in C.CLASSES) +
                     ". The weights are an assumption set by hand, because no dataset has severity labels.")

# ================================================================ survey mode
elif page == "Survey mode":
    st.write(f"Upload many photos from one flight (up to {MAX_SURVEY}). They are ranked by urgency so the worst areas come first.")
    files = st.file_uploader("Aerial photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    coords_file = st.file_uploader("Optional: CSV with columns file, lat, lon to place photos on a map", type=["csv"])
    if files:
        files = files[:MAX_SURVEY]
        raws = [f.getvalue() for f in files]
        imgs = [Image.open(io.BytesIO(b)).convert("RGB") for b in raws]
        with st.spinner(f"Classifying {len(imgs)} photos ..."):
            probs = classify(imgs, [f.name for f in files], raws)
        df = analyse(probs, [f.name for f in files], weights)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Photos", len(df))
        c2.metric("High urgency", int((df["level"] == "High").sum()))
        c3.metric("Medium", int((df["level"] == "Medium").sum()))
        c4.metric("Low confidence (<60%)", int((df["confidence"] < 0.6).sum()))
        want = st.multiselect("Show classes", sorted(df["predicted"].unique()), default=sorted(df["predicted"].unique()))
        view = df[df["predicted"].isin(want)]
        st.dataframe(view[["rank", "file", "predicted", "confidence", "priority", "level"]], width="stretch", hide_index=True)
        st.download_button("Download ranked list (CSV)", df.to_csv(index=False).encode(), "dronerescue_ranked.csv", "text/csv")
        st.markdown("#### Most urgent photos")
        by_name = dict(zip([f.name for f in files], imgs))
        top = df.head(6)
        cols = st.columns(3)
        for i, row in top.iterrows():
            cols[i % 3].image(by_name[row["file"]], width="stretch",
                              caption=f"#{row['rank']} {row['predicted']} | urgency {row['priority']:.0f}")
        st.markdown("#### What the survey contains")
        st.bar_chart(df["predicted"].value_counts())
        if coords_file is not None:
            try:
                geo = pd.read_csv(coords_file).merge(df[["file", "priority"]], on="file")
                geo["size"] = geo["priority"] * 4 + 10
                st.markdown("#### Map")
                st.map(geo.rename(columns={"lat": "latitude", "lon": "longitude"}), latitude="latitude", longitude="longitude",
                       size="size", color="#ff6b4a")
            except Exception as e:                  # noqa: BLE001
                st.error(f"Could not read the coordinates file: {e}")
        else:
            st.caption("No coordinates supplied, so no map is drawn. Public datasets such as AIDERv2 carry no GPS data.")

# ================================================================ model insights
elif page == "Model insights":
    st.subheader("How well does it work?")
    st.caption("All numbers come from the held-out Test split (1,654 images) of the cleaned AIDERv2 dataset. One run per model.")
    table = pd.DataFrame([
        ["Logistic regression (hand-made features)", 0.752, 0.755, 0.751, 0.665, 0.734],
        ["Random forest (hand-made features)", 0.842, 0.854, 0.850, 0.808, 0.821],
        ["CNN trained from scratch (128px)", 0.906, 0.908, 0.901, 0.882, 0.889],
        ["MobileNetV2 fine-tuned (128px)", 0.943, 0.956, 0.953, 0.929, 0.944],
        ["MobileNetV2 fine-tuned (224px)", 0.948, 0.958, 0.958, 0.941, 0.953],
    ], columns=["Model", "Val macro-F1", "Test accuracy", "Test macro-F1", "Weakest-class recall", "Macro-F1 without look-alikes"])
    st.dataframe(table, width="stretch", hide_index=True)
    st.write("**Reading the table.** Recall matters most here, because a missed flood costs more than a false alarm. "
             "The last column scores only the 1,299 Test images with no look-alike in the training data, the most conservative estimate. "
             "With about 1,650 test images, differences of around one point are within noise.")
    shown = False
    for fname, cap in [("final_confusion.png", "Confusion matrix (selected model)"),
                       ("final_reliability.png", "Are the confidence values trustworthy?"),
                       ("final_priority_by_class.png", "Urgency score by true class"),
                       ("final_confident_mistakes.png", "Most confident mistakes"),
                       ("final_gradcam.png", "Where the model looks (Grad-CAM)"),
                       ("phase2_mobilenetv2_224_curves.png", "Training curves, MobileNetV2 224px")]:
        if (OUT / fname).exists():
            st.markdown(f"#### {cap}")
            st.image(str(OUT / fname), width="stretch")
            shown = True
    if not shown:
        st.info("Run `python scripts/08_final_evaluation.py` to add the confusion matrix, calibration and Grad-CAM figures here.")

# ================================================================ about
else:
    st.subheader("About and limits")
    st.markdown("""
**What it does.** It classifies an aerial photo as *earthquake/collapse*, *flood*, *wildfire* or *normal*, and turns the model's
probabilities into an urgency score so many photos can be ranked.

**Data.** AIDERv2 (Zenodo 10891054, CC BY 4.0), 16,723 aerial images in four classes. The dataset had a serious leakage problem:
355 of its 1,654 Test images (21.5%) have a near-identical copy in Train or Val, mostly frames of the same video. Duplicates were
removed from Train and Val (Test was left untouched) and every model is also scored on the Test images without look-alikes.

**Model.** MobileNetV2 with ImageNet weights, a new classification head, and the top layers fine-tuned.

**The urgency score is a rule, not a measurement.** `urgency = 100 x sum(probability x weight)`. The default weights (collapse 1.0,
flood 0.9, fire 0.9, normal 0) are an assumption, because no public dataset has severity labels.

**Limits.**
- The training photos come from web searches, other datasets and video frames, so footage from a real drone may look different and
  the model may do worse on it.
- Each photo gets one label, so a scene with both flood and fire is a weakness.
- Smoke, night, heavy cloud and unusual angles reduce accuracy.
- The model can be confidently wrong. Low-confidence results are flagged, but a high confidence is not a guarantee.
- This is decision support. A person must decide where help goes.
""")
