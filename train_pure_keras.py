import os
# Set backend Keras ke PyTorch (sebelum import keras) agar 100% bebas TensorFlow
os.environ["KERAS_BACKEND"] = "torch"

import json
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# IMPORT 100% MURNI KERAS
import keras
from keras import layers, models, optimizers, callbacks, ops
from keras.applications import MobileNetV2

print(f"[INFO] Versi Keras : {keras.__version__}")
print(f"[INFO] Backend Aktif: {keras.backend.backend()} (Tanpa TensorFlow!)")

# ==========================================
# 1. KONFIGURASI PATH & PARAMETER
# ==========================================
DATASET_DIR = r"C:\project\Fusashield\Chili Leaf Disease Original Dataset"
MODEL_SAVE_PATH = r"C:\project\Fusashield\fusashield_mobilenetv2.keras"
LABELS_SAVE_PATH = r"C:\project\Fusashield\class_names.json"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 25

# ==========================================
# 2. BACA FILE GAMBAR DARI 6 FOLDER LABEL
# ==========================================
class_names = sorted([
    d for d in os.listdir(DATASET_DIR) 
    if os.path.isdir(os.path.join(DATASET_DIR, d))
])

with open(LABELS_SAVE_PATH, "w") as f:
    json.dump(class_names, f, indent=4)

print(f"\n[INFO] 6 Kelas Terdeteksi: {class_names}")

file_paths = []
labels = []

for idx, class_name in enumerate(class_names):
    folder_path = os.path.join(DATASET_DIR, class_name)
    for fname in os.listdir(folder_path):
        if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            file_paths.append(os.path.join(folder_path, fname))
            labels.append(idx)

file_paths = np.array(file_paths)
labels = np.array(labels)

# Bagi data 80% Training dan 20% Validation secara proporsional (stratified)
train_paths, val_paths, train_labels, val_labels = train_test_split(
    file_paths, labels, test_size=0.2, stratify=labels, random_state=42
)

print(f"[INFO] Total Data Training   : {len(train_paths)} gambar")
print(f"[INFO] Total Data Validation : {len(val_paths)} gambar")

# ==========================================
# 3. DATA LOADER MURNI KERAS (keras.utils.PyDataset)
# ==========================================
class ChiliLeafPureKerasDataset(keras.utils.PyDataset):
    def __init__(self, image_paths, labels, batch_size, img_size, num_classes, augment=False, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_classes = num_classes
        self.augment = augment
        self.shuffle = shuffle
        self.indices = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return math.ceil(len(self.image_paths) / self.batch_size)

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_x = []
        batch_y = []

        for i in batch_indices:
            # Load gambar murni dengan Pillow / keras.utils.load_img
            img = keras.utils.load_img(self.image_paths[i], target_size=self.img_size)
            
            # Augmentasi ringan jika mode training aktif
            if self.augment:
                if np.random.rand() > 0.5:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                if np.random.rand() > 0.5:
                    angle = np.random.uniform(-15, 15)
                    img = img.rotate(angle)

            img_array = keras.utils.img_to_array(img)
            batch_x.append(img_array)
            batch_y.append(self.labels[i])

        batch_x = np.array(batch_x, dtype="float32")
        # One-hot encoding murni dengan keras.utils.to_categorical
        batch_y = keras.utils.to_categorical(batch_y, num_classes=self.num_classes)
        return batch_x, batch_y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

train_dataset = ChiliLeafPureKerasDataset(
    train_paths, train_labels, BATCH_SIZE, IMG_SIZE, len(class_names), augment=True, shuffle=True
)
val_dataset = ChiliLeafPureKerasDataset(
    val_paths, val_labels, BATCH_SIZE, IMG_SIZE, len(class_names), augment=False, shuffle=False
)

# ==========================================
# 4. ARSITEKTUR MOBILENETV2 MURNI KERAS
# ==========================================
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # Bekukan bobot ImageNet

inputs = keras.Input(shape=(224, 224, 3), name="input_citra")
# Normalisasi piksel [0, 255] menjadi [-1, 1] sesuai standar MobileNetV2 dengan layer Rescaling murni Keras
x = layers.Rescaling(scale=1./127.5, offset=-1.0, name="normalisasi_mobilenetv2")(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D(name="global_avg_pooling")(x)
x = layers.Dense(128, activation="relu", name="dense_128")(x)
x = layers.Dropout(0.3, name="dropout_30")(x)
outputs = layers.Dense(len(class_names), activation="softmax", name="output_6_label")(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="FusaShield_MobileNetV2")

model.compile(
    optimizer=optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==========================================
# 5. PROSES TRAINING & CALLBACKS KERAS
# ==========================================
keras_callbacks = [
    callbacks.ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),
    callbacks.EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

print("\n[INFO] Memulai Training dengan Keras Murni...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=keras_callbacks
)

# ==========================================
# 6. SIMPAN GRAFIK & CONFUSION MATRIX
# ==========================================
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy", linewidth=2)
plt.plot(history.history["val_accuracy"], label="Val Accuracy", linewidth=2)
plt.title("Akurasi MobileNetV2 (Pure Keras)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss", linewidth=2)
plt.plot(history.history["val_loss"], label="Val Loss", linewidth=2)
plt.title("Loss MobileNetV2 (Pure Keras)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(r"C:\project\Fusashield\grafik_pure_keras.png", dpi=300)
plt.show()

# Evaluasi Confusion Matrix
preds = model.predict(val_dataset, verbose=0)
y_pred = np.argmax(preds, axis=1)
y_true = val_labels

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix - Pure Keras MobileNetV2")
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(r"C:\project\Fusashield\cm_pure_keras.png", dpi=300)
plt.show()