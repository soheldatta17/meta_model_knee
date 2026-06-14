# predict.py — KOA Meta-Ensemble v4.0 inference
# Run in Google Colab. Downloads all models from Drive, loads the saved
# pipeline bundle, and predicts KL grade for every .png in IMAGE_DIR.

from google.colab import drive
drive.mount("/content/drive")

import subprocess, sys

try:
    import gdown
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "gdown", "-q"], check=True)
    import gdown

try:
    from xgboost import XGBClassifier
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "xgboost", "-q"], check=True)

# ── Google Drive asset IDs ────────────────────────────────────────────────────
DRIVE_ASSETS = {
    "model1": {
        "id"   : "1orbyJ0UU44HT3G8inoGstlJ0DhJlQXjj",
        "local": "/content/best_knee_ensemble_cbam.pt",
        "desc" : "Model 1 - CBAM Ensemble (TorchScript) [severity 5-class]",
    },
    "model2": {
        "id"   : "1Hr4gHki9nl6nmXPO0xsAU7FnlfqldHZ8",
        "local": "/content/final_knee_cnn_model.keras",
        "desc" : "Model 2 - Keras CNN + SE Block [JSN regression -> 2-bin]",
    },
    "model3": {
        "id"   : "16ozIZmH36J0K90bY9Jfe4YDTvS2SDDPh",
        "local": "/content/final_mmorphattention.pt",
        "desc" : "Model 3 - MorphAttention (TorchScript) [morph 4-class -> 2]",
    },
    "bundle": {
        "id"   : "1asiAmtlq5t3dcBfv56kUOWQfUlOgaEv8",          # <-- replace with your Drive ID
        "local": "/content/full_pipeline_bundle.pkl",
        "desc" : "Pipeline bundle (XGBoost + scaler + head weights)",
    },
}


# ── IMAGE_DIR: folder containing .png files to predict ───────────────────────
IMAGE_DIR = "/content/images"   # <-- change to your folder path

# ── download helper ───────────────────────────────────────────────────────────
import os

def _gdrive_download(key: str) -> str:
    asset = DRIVE_ASSETS[key]
    local = asset["local"]
    if os.path.exists(local):
        print(f"  [cached] {asset['desc']}  ({os.path.getsize(local)/1e6:.1f} MB)")
        return local
    print(f"  [download] {asset['desc']} ...")
    gdown.download(f"https://drive.google.com/uc?id={asset['id']}", local, quiet=False)
    if not os.path.exists(local):
        raise RuntimeError(f"Download failed for {asset['desc']}. Check Drive share settings.")
    print(f"  saved -> {local}  ({os.path.getsize(local)/1e6:.1f} MB)")
    return local

print("Downloading models ...")
for k in ["model1", "model2", "model3", "bundle"]:
    _gdrive_download(k)

MODEL1_PATH  = DRIVE_ASSETS["model1"]["local"]
MODEL2_PATH  = DRIVE_ASSETS["model2"]["local"]
MODEL3_PATH  = DRIVE_ASSETS["model3"]["local"]
BUNDLE_PATH  = DRIVE_ASSETS["bundle"]["local"]

# ── imports ───────────────────────────────────────────────────────────────────
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import tensorflow as tf

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── load pipeline bundle ──────────────────────────────────────────────────────
print("\nLoading pipeline bundle ...")
with open(BUNDLE_PATH, "rb") as f:
    bundle = pickle.load(f)

xgb            = bundle["xgb"]
scaler         = bundle["scaler"]
CLASS_IDS      = bundle["class_ids"]
SEVERITY_NAMES = bundle["severity_names"]
NUM_CLS        = bundle["num_classes"]
head_cfg       = bundle["head_config"]
print(f"  Bundle v{bundle['version']} loaded — {bundle['description']}")

# ── rebuild KL-grade head from saved weights ──────────────────────────────────
class KLGradeHead(nn.Module):
    def __init__(self, in_dim, hidden, num_cls):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Dropout(0.4),
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, num_cls),
        )
    def forward(self, x):
        return self.net(x)

head = KLGradeHead(head_cfg["in_dim"], head_cfg["hidden"], head_cfg["num_cls"])
head.load_state_dict(bundle["head_state_dict"])
head.eval().to(DEVICE)
print("  KL-grade head restored from bundle.")

# ── load sub-models ───────────────────────────────────────────────────────────
print("\nLoading sub-models ...")
model1 = torch.jit.load(MODEL1_PATH, map_location=DEVICE)
model1.eval()
print("  Model 1 loaded (severity | TorchScript)")

model2_tf = tf.keras.models.load_model(MODEL2_PATH)
model2_tf.trainable = False
_m2_last_act = None
try:
    _m2_last_act = model2_tf.layers[-1].activation.__name__
except Exception:
    pass
MODEL2_HAS_SOFTMAX = (_m2_last_act == "softmax")
print(f"  Model 2 loaded (JSN | Keras | last act='{_m2_last_act}')")

model3 = torch.jit.load(MODEL3_PATH, map_location=DEVICE)
model3.eval()
print("  Model 3 loaded (morphology | TorchScript)")

# ── preprocessing transforms ──────────────────────────────────────────────────
transform_imagenet = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

transform_raw255 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),       # -> [0, 1], multiply by 255 below
])

def preprocess_for_tf(image_path: str):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=1, expand_animations=False)
    img = tf.image.resize(img, [256, 256])
    img = tf.cast(img, tf.float32)
    return tf.expand_dims(img, axis=0)

def jsn_to_2bin(raw_output) -> np.ndarray:
    if MODEL2_HAS_SOFTMAX:
        probs5 = np.array(raw_output[0], dtype=np.float32)
    else:
        probs5 = tf.nn.softmax(raw_output).numpy()[0].astype(np.float32)
    p_narrow = float(probs5[2] + probs5[3] + probs5[4])
    p_normal = float(probs5[0] + probs5[1])
    return np.array([p_narrow, p_normal], dtype=np.float32)

# ── M1 backbone extractor (512D pre-logit embedding) ─────────────────────────
def extract_m1_backbone(img_tensor: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        eff = model1.effnet.features(img_tensor)
        eff = model1.eff_attention(eff)
        eff = model1.effnet.avgpool(eff)
        x1  = torch.flatten(eff, 1)

        r = model1.resnet.conv1(img_tensor)
        r = model1.resnet.bn1(r)
        r = model1.resnet.relu(r)
        r = model1.resnet.maxpool(r)
        r = model1.resnet.layer1(r)
        r = model1.resnet.layer2(r)
        r = model1.resnet.layer3(r)
        r = model1.resnet.layer4(r)
        r = model1.res_attention(r)
        r = model1.resnet.avgpool(r)
        x2 = torch.flatten(r, 1)

        concat = torch.cat([x1, x2], dim=1)   # (1, 2560)
        h = concat
        for name, layer in list(model1.classifier.named_children())[:-1]:
            h = layer(h)
        return h.cpu().numpy()[0]              # (512,)

# ── single-image prediction ───────────────────────────────────────────────────
def predict_image(image_path: str) -> dict:
    img = Image.open(image_path).convert("RGB")

    # M1 — severity probs (5D) + backbone embedding (512D)
    t_in = transform_imagenet(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        sev_probs = F.softmax(model1(t_in), dim=1).cpu().numpy()[0]
    backbone_512 = extract_m1_backbone(t_in)

    # M2 — JSN 2-bin (2D) via Keras model
    img_tf   = preprocess_for_tf(image_path)
    raw_m2   = model2_tf.predict(img_tf, verbose=0)
    jsn_bins = jsn_to_2bin(raw_m2)

    # M3 — morphology 4D, raw [0,255] input
    t_raw = (transform_raw255(img) * 255.0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        morph4 = F.softmax(model3(t_raw), dim=1).cpu().numpy()[0].astype(np.float32)

    # Vfusion — 11D
    vfusion = np.concatenate([sev_probs, jsn_bins, morph4])

    # KL-grade head probs — 5D
    with torch.no_grad():
        bb_t   = torch.tensor(backbone_512[None], dtype=torch.float32).to(DEVICE)
        head_p = F.softmax(head(bb_t), dim=1).cpu().numpy()[0]

    # XGBoost — 16D input
    xgb_input = scaler.transform(np.hstack([vfusion, head_p])[None])
    grade_idx = int(xgb.predict(xgb_input)[0])
    proba     = xgb.predict_proba(xgb_input)[0]
    kl_grade  = CLASS_IDS[grade_idx]

    return {
        "file"         : os.path.basename(image_path),
        "kl_grade"     : kl_grade,
        "severity"     : SEVERITY_NAMES[grade_idx],
        "confidence"   : float(proba[grade_idx]) * 100,
        "probabilities": {
            f"G{CLASS_IDS[i]} {SEVERITY_NAMES[i]}": round(float(proba[i]) * 100, 1)
            for i in range(NUM_CLS)
        },
    }

# ── run on all PNGs in IMAGE_DIR ──────────────────────────────────────────────
png_files = sorted([
    os.path.join(IMAGE_DIR, f)
    for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(".png")
])

if not png_files:
    print(f"\nNo .png files found in {IMAGE_DIR}")
else:
    print(f"\nFound {len(png_files)} PNG file(s). Running inference ...\n")
    print("=" * 60)
    for path in png_files:
        try:
            r = predict_image(path)
            print(f"File       : {r['file']}")
            print(f"KL Grade   : G{r['kl_grade']}  —  {r['severity']}")
            print(f"Confidence : {r['confidence']:.1f}%")
            print("Breakdown  :")
            for label, pct in r["probabilities"].items():
                bar = "#" * int(pct / 2)
                print(f"  {label:<22} {pct:5.1f}%  {bar}")
            print("-" * 60)
        except Exception as e:
            print(f"ERROR on {os.path.basename(path)}: {e}")
            print("-" * 60)
