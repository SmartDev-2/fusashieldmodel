import os
os.environ["KERAS_BACKEND"] = "torch"

import json
import math
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# IMPORT MURNI KERAS
import keras
from keras import layers, optimizers, callbacks
from keras.applications import ResNet50

print(f"[INFO] Versi Keras : {keras.__version__}")
print(f"[INFO] Backend     : {keras.backend.backend()}")

# ==========================================
# 1. KONFIGURASI PATH & PARAMETER (SAMA DENGAN MOBILENETV2)
# ==========================================
DATASET_DIR = r"C:\project\Fusashield\Chili Leaf Disease Original Dataset"
MODEL_SAVE_PATH = r"C:\project\Fusashield\fusashield_resnet50.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 25

class_names = sorted([
    d for d in os.listdir(DATASET_DIR) 
    if os.path.isdir(os.path.join(DATASET_DIR, d))
])

file_paths, labels = [], []
for idx, class_name in enumerate(class_names):
    folder_path = os.path.join(DATASET_DIR, class_name)
    for fname in os.listdir(folder_path):
        if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            file_paths.append(os.path.join(folder_path, fname))
            labels.append(idx)

file_paths = np.array(file_paths)
labels = np.array(labels)

# Split 80:20 dengan seed 42 (data validasi dijamin sama persis dengan MobileNetV2)
train_paths, val_paths, train_labels, val_labels = train_test_split(
    file_paths, labels, test_size=0.2, stratify=labels, random_state=42
)

# ==========================================
# 2. DATA LOADER MURNI KERAS
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
        batch_x, batch_y = [], []

        for i in batch_indices:
            img = keras.utils.load_img(self.image_paths[i], target_size=self.img_size)
            if self.augment:
                if np.random.rand() > 0.5:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                if np.random.rand() > 0.5:
                    img = img.rotate(np.random.uniform(-15, 15))

            batch_x.append(keras.utils.img_to_array(img))
            batch_y.append(self.labels[i])

        batch_x = np.array(batch_x, dtype="float32")
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
# 3. ARSITEKTUR RESNET-50 MURNI KERAS
# ==========================================
base_model = ResNet50(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

inputs = keras.Input(shape=(224, 224, 3), name="input_citra")
# Normalisasi piksel [-1, 1] agar adil dengan skema pengujian kita
x = layers.Rescaling(scale=1./127.5, offset=-1.0, name="normalisasi_resnet")(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D(name="global_avg_pooling")(x)
x = layers.Dense(128, activation="relu", name="dense_128")(x)
x = layers.Dropout(0.3, name="dropout_30")(x)
outputs = layers.Dense(len(class_names), activation="softmax", name="output_6_label")(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="FusaShield_ResNet50")

model.compile(
    optimizer=optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==========================================
# 4. TRAINING & PENCATATAN WAKTU KOMPUTASI
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

print("\n[INFO] Memulai Training ResNet-50...")
start_time = time.time()

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=keras_callbacks
)

durasi_menit = (time.time() - start_time) / 60.0
print(f"\n[INFO] Total Waktu Training ResNet-50: {durasi_menit:.2f} menit")

# ==========================================
# 5. SIMPAN GRAFIK & CONFUSION MATRIX RESNET-50
# ==========================================
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy", linewidth=2)
plt.plot(history.history["val_accuracy"], label="Val Accuracy", linewidth=2)
plt.title("Akurasi ResNet-50 (Pure Keras)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss", linewidth=2)
plt.plot(history.history["val_loss"], label="Val Loss", linewidth=2)
plt.title("Loss ResNet-50 (Pure Keras)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(r"C:\project\Fusashield\grafik_resnet50_keras.png", dpi=300)
plt.show()

preds = model.predict(val_dataset, verbose=0)
y_pred = np.argmax(preds, axis=1)

print("\n=== CLASSIFICATION REPORT RESNET-50 ===")
print(classification_report(val_labels, y_pred, target_names=class_names))

cm = confusion_matrix(val_labels, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix - Pure Keras ResNet-50")
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(r"C:\project\Fusashield\cm_resnet50_keras.png", dpi=300)
plt.show()