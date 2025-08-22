import os
import subprocess
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# --- CONFIG ---
IMAGE_SIZE = (640, 480)
BATCH_SIZE = 16
EPOCHS = 10
K_FOLDS = 5
DATA_DIR = "AI_data"
CLASSES = ['bursts', 'noise']

# --- 0. SYNC FILES FROM GOOGLE DRIVE VIA RCLONE ---
print("☁️  Syncing data from Google Drive using rclone...")

# Replace 'gdrive' with your rclone remote name
RCLONE_REMOTE = "gdrive"
REMOTE_BASE = "MyDrive/data/processing/AI_data"
LOCAL_BASE = DATA_DIR

# Create local data directories
os.makedirs(LOCAL_BASE, exist_ok=True)

for label in CLASSES:
    remote_path = f"{RCLONE_REMOTE}:{REMOTE_BASE}/{label}"
    local_path = os.path.join(LOCAL_BASE, label)
    os.makedirs(local_path, exist_ok=True)

    print(f"📂 Syncing {label} data...")
    result = subprocess.run(
        ["rclone", "copy", remote_path, local_path, "--progress"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        print(f"❌ Failed to sync {label}: {result.stderr}")
    else:
        print(f"✅ Synced {label} successfully.")

# --- 1. CREATE DATAFRAME ---
filepaths = []
labels = []

for label in CLASSES:
    folder = os.path.join(DATA_DIR, label)
    for fname in os.listdir(folder):
        if fname.endswith(".png"):
            filepaths.append(os.path.join(folder, fname))
            labels.append(label)

df = pd.DataFrame({
    'filepath': filepaths,
    'label': labels
})
df['label_idx'] = df['label'].map({label: i for i, label in enumerate(CLASSES)})

# --- 2. K-FOLD CROSS-VALIDATION (Evaluation Only) ---
skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
fold_accuracies = []

print("\n🔍 Evaluating model with cross-validation...\n")

for fold, (train_idx, val_idx) in enumerate(skf.split(df['filepath'], df['label_idx'])):
    print(f"🌀 Fold {fold + 1}/{K_FOLDS}")

    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()

    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2]
    )
    val_gen = ImageDataGenerator(rescale=1./255)

    train_data = train_gen.flow_from_dataframe(
        train_df,
        x_col='filepath',
        y_col='label',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=True
    )
    val_data = val_gen.flow_from_dataframe(
        val_df,
        x_col='filepath',
        y_col='label',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    y_train_labels = train_df['label_idx'].values
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_labels),
        y=y_train_labels
    )
    class_weights = dict(enumerate(class_weights))

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[1], IMAGE_SIZE[0], 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS,
        class_weight=class_weights,
        verbose=0  # Quiet training
    )

    val_acc = history.history['val_accuracy'][-1]
    fold_accuracies.append(val_acc)
    print(f"✅ Fold {fold + 1} validation accuracy: {val_acc:.4f}")

# --- 3. Print Evaluation Summary ---
mean_acc = np.mean(fold_accuracies)
print(f"\n📊 Cross-validation complete. Mean validation accuracy: {mean_acc:.4f}\n")

# --- 4. FINAL TRAINING ON FULL DATASET ---
print("🚀 Training final model on the entire dataset...")

# Recreate image generator for full data
full_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

full_data = full_gen.flow_from_dataframe(
    df,
    x_col='filepath',
    y_col='label',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

# Compute final class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(df['label_idx']),
    y=df['label_idx']
)
class_weights = dict(enumerate(class_weights))

# Create new model
final_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[1], IMAGE_SIZE[0], 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
final_model.compile(optimizer=Adam(),
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

# Train final model
final_model.fit(
    full_data,
    epochs=EPOCHS,
    class_weight=class_weights,
    verbose=2
)

# --- 5. SAVE FINAL MODEL ---
final_model.save("final_model.h5")
print("💾 Final model saved as: final_model.h5")
