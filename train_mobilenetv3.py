import os
# Set backend Keras ke PyTorch (sebelum import keras)
os.environ["KERAS_BACKEND"] = "torch"

import json
import math
import time
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

# IMPORT MURNI KERAS 3
import keras
from keras import layers, models, optimizers, callbacks
from keras.applications import MobileNetV3Large

print(f"[INFO] Versi Keras : {keras.__version__}")
print(f"[INFO] Backend Aktif: {keras.backend.backend()} (Tanpa TensorFlow runtime!)")

# ==========================================
# 1. KONFIGURASI PATH & PARAMETER
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "Chili Leaf Disease Original Dataset")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "fusashield_mobilenetv3.keras")
LABELS_SAVE_PATH = os.path.join(BASE_DIR, "class_names.json")

# Folder output tabel evaluasi
OUTPUT_DIR = os.path.join(BASE_DIR, "hasil_tabel_evaluasi")
os.makedirs(OUTPUT_DIR, exist_ok=True)

REPORT_SAVE_PATH = os.path.join(OUTPUT_DIR, "report_mobilenetv3.txt")
GRAFIK_SAVE_PATH = os.path.join(OUTPUT_DIR, "grafik_mobilenetv3_keras.png")
CM_SAVE_PATH = os.path.join(OUTPUT_DIR, "cm_mobilenetv3_keras.png")

# File tabel evaluasi (CSV & PNG)
TABEL_METRIK_CSV = os.path.join(OUTPUT_DIR, "tabel_metrik_mobilenetv3.csv")
TABEL_METRIK_PNG = os.path.join(OUTPUT_DIR, "tabel_metrik_mobilenetv3.png")
TABEL_RINGKASAN_CSV = os.path.join(OUTPUT_DIR, "tabel_ringkasan_mobilenetv3.csv")
TABEL_RINGKASAN_PNG = os.path.join(OUTPUT_DIR, "tabel_ringkasan_mobilenetv3.png")

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 25
LEARNING_RATE = 0.0001

# ==========================================
# 2. BACA FILE GAMBAR DARI DATASET
# ==========================================
if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(f"[ERROR] Folder dataset tidak ditemukan di: {DATASET_DIR}")

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

# Split 80% Training dan 20% Validation secara stratified
train_paths, val_paths, train_labels, val_labels = train_test_split(
    file_paths, labels, test_size=0.2, stratify=labels, random_state=42
)

print(f"[INFO] Total Data Training   : {len(train_paths)} gambar")
print(f"[INFO] Total Data Validation : {len(val_paths)} gambar")

# ==========================================
# 3. DATA LOADER (keras.utils.PyDataset)
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
            img = keras.utils.load_img(self.image_paths[i], target_size=self.img_size)
            
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
# 4. ARSITEKTUR MOBILENETV3 LARGE
# ==========================================
base_model = MobileNetV3Large(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
    pooling="avg",
    include_preprocessing=True
)
base_model.trainable = False

inputs = keras.Input(shape=(224, 224, 3), name="input_citra")
x = base_model(inputs, training=False)
x = layers.Dense(128, activation="relu", name="dense_128")(x)
x = layers.Dropout(0.3, name="dropout_30")(x)
outputs = layers.Dense(len(class_names), activation="softmax", name="output_6_label")(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="FusaShield_MobileNetV3")

model.compile(
    optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Hitung jumlah parameter
total_params = model.count_params()
trainable_params = sum(p.numel() for p in model.trainable_weights)
non_trainable_params = sum(p.numel() for p in model.non_trainable_weights)

# ==========================================
# 5. TRAINING & CALLBACKS
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

print("\n[INFO] Memulai Training MobileNetV3...")
start_train_time = time.time()
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=keras_callbacks
)
total_train_duration = time.time() - start_train_time

# ==========================================
# 6. SIMPAN GRAFIK TRAINING
# ==========================================
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy", linewidth=2)
plt.plot(history.history["val_accuracy"], label="Val Accuracy", linewidth=2)
plt.title("Akurasi MobileNetV3 (Pure Keras)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss", linewidth=2)
plt.plot(history.history["val_loss"], label="Val Loss", linewidth=2)
plt.title("Loss MobileNetV3 (Pure Keras)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(GRAFIK_SAVE_PATH, dpi=300)
plt.close()
print(f"[INFO] Grafik akurasi & loss tersimpan di: {GRAFIK_SAVE_PATH}")

# ==========================================
# 7. EVALUASI METRIK, SPEED TEST & CONFUSION MATRIX
# ==========================================
print("\n[INFO] Mengevaluasi model terbaik...")
best_model = keras.models.load_model(MODEL_SAVE_PATH)

# Benchmarking waktu inferensi rata-rata per citra
print("[INFO] Mengukur waktu inferensi...")
sample_img = np.random.rand(1, 224, 224, 3).astype("float32")
for _ in range(5):  # Warmup
    _ = best_model.predict(sample_img, verbose=0)

latencies = []
for _ in range(50):
    t0 = time.time()
    _ = best_model.predict(sample_img, verbose=0)
    latencies.append((time.time() - t0) * 1000)
avg_latency_ms = np.mean(latencies)

# Prediksi seluruh data validasi
preds = best_model.predict(val_dataset, verbose=0)
y_pred = np.argmax(preds, axis=1)
y_true = val_labels

# Classification report
rep_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
rep_text = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print("\n=== CLASSIFICATION REPORT ===")
print(rep_text)

with open(REPORT_SAVE_PATH, "w") as f:
    f.write(rep_text)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
if HAS_SEABORN:
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
else:
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=30, ha="right")
    plt.yticks(tick_marks, class_names)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
plt.title("Confusion Matrix - MobileNetV3")
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(CM_SAVE_PATH, dpi=300)
plt.close()

# Simpan CM ke CSV
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
cm_df.to_csv(os.path.join(OUTPUT_DIR, "3c_tabel_cm_mobilenetv3.csv"))

# ==========================================
# 8. GENERATE TABEL HASIL EVALUASI (CSV & GAMBAR PNG)
# ==========================================
print("\n[INFO] Menghasilkan Tabel Evaluasi...")

def export_df_to_png(df, output_path, title, header_color="#2b579a"):
    fig, ax = plt.subplots(figsize=(max(8, len(df.columns) * 2), max(3, len(df) * 0.45 + 1.2)))
    ax.axis("off")
    ax.axis("tight")
    
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.4)
    
    # Styling cell
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", weight="bold")
        else:
            if i % 2 == 0:
                cell.set_facecolor("#f2f5f9")
            else:
                cell.set_facecolor("white")
            if i == len(df):  # Baris total / summary
                cell.set_text_props(weight="bold")
    
    plt.title(title, fontsize=12, weight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

# --- Tabel 1: Metrik Per Kelas ---
kelas_data = []
for c in class_names:
    row = {
        "Kelas Daun Cabai": c,
        "Jml Uji": int(rep_dict[c]["support"]),
        "Precision": f"{rep_dict[c]['precision'] * 100:.2f}%",
        "Recall": f"{rep_dict[c]['recall'] * 100:.2f}%",
        "F1-Score": f"{rep_dict[c]['f1-score'] * 100:.2f}%"
    }
    kelas_data.append(row)

# Tambahkan baris Macro Average
kelas_data.append({
    "Kelas Daun Cabai": "RATA-RATA (MACRO AVG)",
    "Jml Uji": len(y_true),
    "Precision": f"{rep_dict['macro avg']['precision'] * 100:.2f}%",
    "Recall": f"{rep_dict['macro avg']['recall'] * 100:.2f}%",
    "F1-Score": f"{rep_dict['macro avg']['f1-score'] * 100:.2f}%"
})

df_per_kelas = pd.DataFrame(kelas_data)
df_per_kelas.to_csv(TABEL_METRIK_CSV, index=False)
export_df_to_png(df_per_kelas, TABEL_METRIK_PNG, "Tabel Evaluasi Metrik Per Kelas - MobileNetV3 (FusaShield)", "#1e7e34")
print(f"  -> Tabel Metrik Per Kelas : {TABEL_METRIK_CSV} & .png")

# --- Tabel 2: Ringkasan Performa Keseluruhan ---
train_acc = max(history.history["accuracy"]) * 100
val_acc = max(history.history["val_accuracy"]) * 100
train_loss = min(history.history["loss"])
val_loss = min(history.history["val_loss"])
model_file_size_mb = os.path.getsize(MODEL_SAVE_PATH) / (1024 * 1024)

ringkasan_data = [
    {"Parameter Evaluasi": "Akurasi Training (%)", "Nilai": f"{train_acc:.2f}%"},
    {"Parameter Evaluasi": "Akurasi Validasi (%)", "Nilai": f"{val_acc:.2f}%"},
    {"Parameter Evaluasi": "Selisih Train-Val (Gap)", "Nilai": f"{abs(train_acc - val_acc):.2f}%"},
    {"Parameter Evaluasi": "Loss Training Akhir", "Nilai": f"{train_loss:.4f}"},
    {"Parameter Evaluasi": "Loss Validasi Akhir", "Nilai": f"{val_loss:.4f}"},
    {"Parameter Evaluasi": "Precision (Macro Average)", "Nilai": f"{rep_dict['macro avg']['precision'] * 100:.2f}%"},
    {"Parameter Evaluasi": "Recall (Macro Average)", "Nilai": f"{rep_dict['macro avg']['recall'] * 100:.2f}%"},
    {"Parameter Evaluasi": "F1-Score (Macro Average)", "Nilai": f"{rep_dict['macro avg']['f1-score'] * 100:.2f}%"},
    {"Parameter Evaluasi": "F1-Score (Weighted Average)", "Nilai": f"{rep_dict['weighted avg']['f1-score'] * 100:.2f}%"},
    {"Parameter Evaluasi": "Total Parameter Jaringan", "Nilai": f"{total_params:,}"},
    {"Parameter Evaluasi": "Parameter Trainable", "Nilai": f"{trainable_params:,}"},
    {"Parameter Evaluasi": "Parameter Non-Trainable (Frozen)", "Nilai": f"{non_trainable_params:,}"},
    {"Parameter Evaluasi": "Ukuran File Model (.keras)", "Nilai": f"{model_file_size_mb:.2f} MB"},
    {"Parameter Evaluasi": "Waktu Inferensi Rata-rata", "Nilai": f"{avg_latency_ms:.2f} ms / citra"},
    {"Parameter Evaluasi": "Total Durasi Training", "Nilai": f"{total_train_duration / 60:.1f} menit"}
]

df_ringkasan = pd.DataFrame(ringkasan_data)
df_ringkasan.to_csv(TABEL_RINGKASAN_CSV, index=False)
export_df_to_png(df_ringkasan, TABEL_RINGKASAN_PNG, "Tabel Ringkasan Performa MobileNetV3 (FusaShield)", "#17a2b8")
print(f"  -> Tabel Ringkasan Model  : {TABEL_RINGKASAN_CSV} & .png")

print("\n" + "=" * 60)
print("✅ TRAINING & GENERASI TABEL EVALUASI SELESAI!")
print("=" * 60)
