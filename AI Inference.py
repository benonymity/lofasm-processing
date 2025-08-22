import os
import subprocess
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input

# --- CONFIG ---
IMAGE_SIZE = (640, 480)
MODEL_PATH = "final_model.h5"
RCLONE_REMOTE = "gdrive"
REMOTE_DIRS = [
    "MyDrive/data/processing/new_processed_data/log",
    "MyDrive/data/processing/new_processed_data/vmax"
]
LOCAL_DOWNLOAD_DIR = "inference_data"
OUTPUT_CSV = "inference_results.csv"
CLASSES = ['noise', 'burst']  # 0 = noise, 1 = burst

# --- 1. Download New Data with rclone ---
print("☁️  Downloading new data from Google Drive...")

os.makedirs(LOCAL_DOWNLOAD_DIR, exist_ok=True)

for remote_subdir in REMOTE_DIRS:
    remote_path = f"{RCLONE_REMOTE}:{remote_subdir}"
    local_subdir = os.path.join(LOCAL_DOWNLOAD_DIR, os.path.basename(remote_subdir))
    os.makedirs(local_subdir, exist_ok=True)

    print(f"📂 Pulling from: {remote_path}")
    result = subprocess.run(
        ["rclone", "copy", remote_path, local_subdir, "--progress"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if result.returncode != 0:
        print(f"❌ Failed to sync {remote_subdir}: {result.stderr}")
    else:
        print(f"✅ Synced {remote_subdir} successfully.")

# --- 2. Load Model ---
print(f"\n📦 Loading trained model from: {MODEL_PATH}")
model = load_model(MODEL_PATH)

# --- 3. Collect Images for Inference ---
image_paths = []
sources = []

for subfolder in os.listdir(LOCAL_DOWNLOAD_DIR):
    subfolder_path = os.path.join(LOCAL_DOWNLOAD_DIR, subfolder)
    for fname in os.listdir(subfolder_path):
        if fname.endswith(".png"):
            image_paths.append(os.path.join(subfolder_path, fname))
            sources.append(subfolder)

print(f"🔍 Found {len(image_paths)} PNG files for inference.")

# --- 4. Preprocess & Predict ---
def preprocess_image(img_path):
    img = load_img(img_path, target_size=IMAGE_SIZE)
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Rescale to [0, 1]
    return img_array

print("🧠 Running inference...")
X = np.array([preprocess_image(path) for path in image_paths])
pred_probs = model.predict(X, verbose=0).flatten()
pred_labels = (pred_probs >= 0.5).astype(int)

# --- 5. Save Predictions ---
results_df = pd.DataFrame({
    "filepath": image_paths,
    "source_folder": sources,
    "predicted_class": [CLASSES[i] for i in pred_labels],
    "confidence": pred_probs
})

results_df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Inference complete. Results saved to: {OUTPUT_CSV}")