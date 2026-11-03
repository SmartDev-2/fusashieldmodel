import os
os.environ["KERAS_BACKEND"] = "torch"  # Murni menggunakan PyTorch Backend (Tanpa TensorFlow)

import json
import time
import numpy as np
import pandas as pd
from PIL import Image
import keras
from keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt

# ==========================================================
# 1. KONFIGURASI DIREKTORI & AUTO-DETECT DATASET
# ==========================================================
BASE_PROJECT_DIR = r"C:\project\Fusashield"
OUTPUT_DIR = os.path.join(BASE_PROJECT_DIR, "hasil_tabel_evaluasi")

PATH_MOBILENET = os.path.join(BASE_PROJECT_DIR, "fusashield_mobilenetv2.keras")
PATH_RESNET = os.path.join(BASE_PROJECT_DIR, "fusashield_resnet50.keras")

def cari_folder_dataset(root_folder):
    ekstensi_gambar = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    for root, dirs, files in os.walk(root_folder):
        if "hasil_tabel" in root or "venv" in root or ".git" in root:
            continue
        if len(dirs) >= 5:
            jml_sub = 0
            for d in dirs:
                try:
                    isi = os.listdir(os.path.join(root, d))
                    if any(f.lower().endswith(ekstensi_gambar) for f in isi):
                        jml_sub += 1
                except Exception:
                    pass
            if jml_sub >= 5:
                return root
    return os.path.join(root_folder, "dataset")

DATASET_DIR =  r"C:\project\Fusashield\Chili Leaf Disease Original Dataset"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1e-4
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================================
# 2. LOAD DATASET MURNI NUMPY + PIL (TANPA TENSORFLOW!)
# ==========================================================
def muat_dataset_tanpa_tf(dataset_dir, img_size=(224, 224), val_split=0.20, seed=42):
    print(f"[INFO] Membaca dataset dari: {dataset_dir}")
    ekstensi_valid = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    
    class_names = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])
    num_classes = len(class_names)
    
    X_list = []
    y_indices = []
    
    for idx_kelas, nama_kelas in enumerate(class_names):
        folder_kelas = os.path.join(dataset_dir, nama_kelas)
        daftar_file = sorted(os.listdir(folder_kelas))
        jml_per_kelas = 0
        for nama_file in daftar_file:
            if nama_file.lower().endswith(ekstensi_valid):
                path_img = os.path.join(folder_kelas, nama_file)
                try:
                    with Image.open(path_img) as img:
                        img_rgb = img.convert("RGB").resize(img_size, Image.Resampling.BILINEAR)
                        X_list.append(np.asarray(img_rgb, dtype=np.float32))
                        y_indices.append(idx_kelas)
                        jml_per_kelas += 1
                except Exception as e:
                    print(f"  [WARN] Gagal membaca {path_img}: {e}")
        print(f"  -> Kelas [{idx_kelas}] {nama_kelas:<28}: {jml_per_kelas} citra")

    X_all = np.array(X_list, dtype=np.float32)
    y_all_idx = np.array(y_indices, dtype=np.int64)
    
    # Stratified Split 80% Train : 20% Validasi agar proporsi tiap kelas seimbang
    X_train, X_val, y_train_idx, y_val_idx = train_test_split(
        X_all, y_all_idx,
        test_size=val_split,
        random_state=seed,
        stratify=y_all_idx
    )
    
    # Konversi label ke One-Hot Encoding (Categorical) untuk 6 kelas
    y_train = keras.utils.to_categorical(y_train_idx, num_classes=num_classes)
    y_val = keras.utils.to_categorical(y_val_idx, num_classes=num_classes)
    
    print(f"[INFO] Total Citra: {len(X_all)} | Train (80%): {len(X_train)} | Validasi (20%): {len(X_val)}")
    return X_train, y_train, X_val, y_val, y_val_idx, class_names

X_train, y_train, X_val, y_val, y_val_true_idx, class_names = muat_dataset_tanpa_tf(
    DATASET_DIR, IMG_SIZE, val_split=0.20, seed=SEED
)
num_classes = len(class_names)


# ==========================================================
# 3. FUNGSI PEMBANGUN ARSITEKTUR MODEL (APPLE-TO-APPLE)
# ==========================================================
def bangun_model(arsitektur, num_classes):
    inputs = keras.Input(shape=(224, 224, 3), name="input_citra")
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="rescaling_minus1_to_1")(inputs)

    if arsitektur == "MobileNetV2":
        base_model = keras.applications.MobileNetV2(
            input_shape=(224, 224, 3), include_top=False, weights="imagenet"
        )
    elif arsitektur == "ResNet50":
        base_model = keras.applications.ResNet50(
            input_shape=(224, 224, 3), include_top=False, weights="imagenet"
        )
    else:
        raise ValueError("Pilih 'MobileNetV2' atau 'ResNet50'")

    base_model.trainable = False
    x = base_model(x, training=False)

    # Head Identik
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(128, activation="relu", name="dense_128")(x)
    x = layers.Dropout(0.3, name="dropout_03")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output_softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=f"FusaShield_{arsitektur}")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model, len(base_model.layers)


# ==========================================================
# 4. FUNGSI RENDER TABEL PANDAS KE GAMBAR PNG (SIAP PPT)
# ==========================================================
def simpan_tabel_ke_png(df, judul, nama_file, lebar_fig=12, tinggi_baris=0.45):
    tinggi_fig = max(3.5, (len(df) + 2) * tinggi_baris)
    fig, ax = plt.subplots(figsize=(lebar_fig, tinggi_fig))
    ax.axis("off")
    ax.set_title(judul, fontsize=13, fontweight="bold", pad=15, color="#1b4332")

    tabel = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center"
    )
    tabel.auto_set_font_size(False)
    tabel.set_fontsize(9.5)
    tabel.scale(1.0, 1.45)

    for (row, col), cell in tabel.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#2d6a4f")
            cell.set_edgecolor("white")
        else:
            cell.set_facecolor("#f8f9fa" if row % 2 == 1 else "#e9ecef")
            cell.set_edgecolor("#dee2e6")
            if col == 0:
                cell.set_text_props(weight="bold")

    plt.tight_layout()
    path_png = os.path.join(OUTPUT_DIR, nama_file)
    plt.savefig(path_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"       -> [GAMBAR TABEL DISIMPAN]: {path_png}")


# ==========================================================
# 5. EKSEKUSI EVALUASI LENGKAP PER MODEL
# ==========================================================
def jalankan_evaluasi_model(nama_model, path_model, paksa_train=False):
    path_hist_json = os.path.join(OUTPUT_DIR, f"history_{nama_model.lower()}.json")
    layer_backbone = 154 if nama_model == "MobileNetV2" else 175
    history_dict = None

    if os.path.exists(path_model) and not paksa_train:
        print(f"\n[LOAD] Memuat bobot {nama_model} dari: {path_model}")
        model = keras.models.load_model(path_model)
        if os.path.exists(path_hist_json):
            with open(path_hist_json, "r") as f:
                history_dict = json.load(f)
    else:
        print(f"\n[TRAIN] Melatih {nama_model} selama {EPOCHS} epoch...")
        model, layer_backbone = bangun_model(nama_model, num_classes)
        hist = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            verbose=1
        )
        model.save(path_model)
        history_dict = hist.history
        with open(path_hist_json, "w") as f:
            json.dump(history_dict, f, indent=4)

    print(f"[EVAL] Menghitung metrik eksak {nama_model} pada data Train & Validasi...")
    train_loss, train_acc = model.evaluate(X_train, y_train, batch_size=BATCH_SIZE, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, batch_size=BATCH_SIZE, verbose=0)

    # Kumpulkan prediksi per kelas & waktu inferensi
    t0 = time.time()
    preds = model.predict(X_val, batch_size=BATCH_SIZE, verbose=0)
    waktu_total = time.time() - t0
    ms_per_citra = (waktu_total / len(X_val)) * 1000.0

    y_pred = np.argmax(preds, axis=1)
    y_true = y_val_true_idx

    # Metrik Global (Macro & Weighted)
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_wei, r_wei, f1_wei, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Metrik Per Kelas (6 Kelas)
    p_cls, r_cls, f1_cls, sup_cls = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(num_classes), zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    total_params = model.count_params()
    trainable_params = sum(keras.ops.size(w).item() for w in model.trainable_weights)
    non_trainable_params = total_params - trainable_params
    ukuran_mb = os.path.getsize(path_model) / (1024 * 1024)

    print(f"\n--- Classification Report: {nama_model} ---")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0))

    return {
        "nama": nama_model,
        "train_acc_pct": round(float(train_acc) * 100, 2),
        "train_loss": round(float(train_loss), 4),
        "val_acc_pct": round(float(val_acc) * 100, 2),
        "val_loss": round(float(val_loss), 4),
        "gap_acc_pct": round(abs(float(train_acc) - float(val_acc)) * 100, 2),
        "precision_macro_pct": round(float(p_mac) * 100, 2),
        "recall_macro_pct": round(float(r_mac) * 100, 2),
        "f1_macro_pct": round(float(f1_mac) * 100, 2),
        "f1_weighted_pct": round(float(f1_wei) * 100, 2),
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "non_trainable_params": int(non_trainable_params),
        "ukuran_mb": round(float(ukuran_mb), 2),
        "layer_backbone": int(layer_backbone),
        "inferensi_ms": round(float(ms_per_citra), 2),
        "per_kelas": {
            "precision": [round(float(x) * 100, 2) for x in p_cls],
            "recall": [round(float(x) * 100, 2) for x in r_cls],
            "f1": [round(float(x) * 100, 2) for x in f1_cls],
            "support": [int(x) for x in sup_cls],
        },
        "cm": cm,
        "history": history_dict
    }


# ==========================================================
# 6. PEMBUATAN 4 TABEL LENGKAP (TERMINAL, CSV, DAN GAMBAR PNG)
# ==========================================================
if __name__ == "__main__":
    mnet = jalankan_evaluasi_model("MobileNetV2", PATH_MOBILENET, paksa_train=False)
    rnet = jalankan_evaluasi_model("ResNet50", PATH_RESNET, paksa_train=False)

    # TABEL 1: PERBANDINGAN UTAMA & SPESIFIKASI ARSITEKTUR
    data_tabel_1 = [
        ["Akurasi Training (%)", f"{mnet['train_acc_pct']:.2f}%", f"{rnet['train_acc_pct']:.2f}%", f"{mnet['train_acc_pct'] - rnet['train_acc_pct']:+.2f}%"],
        ["Akurasi Validasi (%)", f"{mnet['val_acc_pct']:.2f}%", f"{rnet['val_acc_pct']:.2f}%", f"{mnet['val_acc_pct'] - rnet['val_acc_pct']:+.2f}%"],
        ["Selisih Train-Val (Indikator Overfit)", f"{mnet['gap_acc_pct']:.2f}%", f"{rnet['gap_acc_pct']:.2f}%", "Berimpit / Stabil" if mnet['gap_acc_pct'] < 3.0 else "Waspada"],
        ["Loss Training Akhir", f"{mnet['train_loss']:.4f}", f"{rnet['train_loss']:.4f}", f"{mnet['train_loss'] - rnet['train_loss']:+.4f}"],
        ["Loss Validasi Akhir", f"{mnet['val_loss']:.4f}", f"{rnet['val_loss']:.4f}", f"{mnet['val_loss'] - rnet['val_loss']:+.4f}"],
        ["Precision (Macro Average %)", f"{mnet['precision_macro_pct']:.2f}%", f"{rnet['precision_macro_pct']:.2f}%", f"{mnet['precision_macro_pct'] - rnet['precision_macro_pct']:+.2f}%"],
        ["Recall (Macro Average %)", f"{mnet['recall_macro_pct']:.2f}%", f"{rnet['recall_macro_pct']:.2f}%", f"{mnet['recall_macro_pct'] - rnet['recall_macro_pct']:+.2f}%"],
        ["F1-Score (Macro Average %)", f"{mnet['f1_macro_pct']:.2f}%", f"{rnet['f1_macro_pct']:.2f}%", f"{mnet['f1_macro_pct'] - rnet['f1_macro_pct']:+.2f}%"],
        ["F1-Score (Weighted Average %)", f"{mnet['f1_weighted_pct']:.2f}%", f"{rnet['f1_weighted_pct']:.2f}%", f"{mnet['f1_weighted_pct'] - rnet['f1_weighted_pct']:+.2f}%"],
        ["Total Parameter Jaringan", f"{mnet['total_params']:,}", f"{rnet['total_params']:,}", f"{rnet['total_params'] / mnet['total_params']:.1f}x Lebih Ringan"],
        ["Parameter Trainable (Head)", f"{mnet['trainable_params']:,}", f"{rnet['trainable_params']:,}", "Head Identik"],
        ["Parameter Non-Trainable (Frozen)", f"{mnet['non_trainable_params']:,}", f"{rnet['non_trainable_params']:,}", "Backbone Beku"],
        ["Jumlah Layer Backbone", f"{mnet['layer_backbone']} Layer", f"{rnet['layer_backbone']} Layer", "-21 Layer"],
        ["Ukuran File Model (.keras)", f"{mnet['ukuran_mb']:.2f} MB", f"{rnet['ukuran_mb']:.2f} MB", f"Hemat {rnet['ukuran_mb'] - mnet['ukuran_mb']:.2f} MB"],
        ["Waktu Inferensi Rata-rata / Citra", f"{mnet['inferensi_ms']:.2f} ms", f"{rnet['inferensi_ms']:.2f} ms", f"{rnet['inferensi_ms'] / max(mnet['inferensi_ms'], 0.01):.1f}x Lebih Cepat"]
    ]
    df_tabel_1 = pd.DataFrame(
        data_tabel_1,
        columns=["Parameter Evaluasi", "MobileNetV2", "ResNet-50", "Selisih / Keterangan"]
    )

    print("\n" + "=" * 95)
    print("TABEL 1: RINGKASAN PERBANDINGAN PERFORMA & ARSITEKTUR (MOBILENETV2 VS RESNET-50)")
    print("=" * 95)
    print(df_tabel_1.to_string(index=False))
    df_tabel_1.to_csv(os.path.join(OUTPUT_DIR, "1_tabel_perbandingan_utama.csv"), index=False)
    simpan_tabel_ke_png(
        df_tabel_1,
        "Tabel 1. Perbandingan Eksak Performa & Arsitektur: MobileNetV2 vs ResNet-50",
        "1_tabel_perbandingan_utama.png",
        lebar_fig=13
    )

    # TABEL 2: PERBANDINGAN METRIK PER KELAS (6 KELAS KELUARAN)
    data_tabel_2 = []
    for i, nama_kls in enumerate(class_names):
        data_tabel_2.append([
            nama_kls,
            mnet["per_kelas"]["support"][i],
            f"{mnet['per_kelas']['precision'][i]:.2f}%",
            f"{mnet['per_kelas']['recall'][i]:.2f}%",
            f"{mnet['per_kelas']['f1'][i]:.2f}%",
            f"{rnet['per_kelas']['precision'][i]:.2f}%",
            f"{rnet['per_kelas']['recall'][i]:.2f}%",
            f"{rnet['per_kelas']['f1'][i]:.2f}%",
            f"{mnet['per_kelas']['f1'][i] - rnet['per_kelas']['f1'][i]:+.2f}%"
        ])

    data_tabel_2.append([
        "RATA-RATA (MACRO AVG)",
        sum(mnet["per_kelas"]["support"]),
        f"{mnet['precision_macro_pct']:.2f}%",
        f"{mnet['recall_macro_pct']:.2f}%",
        f"{mnet['f1_macro_pct']:.2f}%",
        f"{rnet['precision_macro_pct']:.2f}%",
        f"{rnet['recall_macro_pct']:.2f}%",
        f"{rnet['f1_macro_pct']:.2f}%",
        f"{mnet['f1_macro_pct'] - rnet['f1_macro_pct']:+.2f}%"
    ])

    df_tabel_2 = pd.DataFrame(
        data_tabel_2,
        columns=[
            "Kelas Daun Cabai", "Jml Uji",
            "MNet Precision", "MNet Recall", "MNet F1",
            "ResNet Precision", "ResNet Recall", "ResNet F1",
            "Selisih F1"
        ]
    )

    print("\n" + "=" * 110)
    print("TABEL 2: EVALUASI DETAIL PER KELAS (PRECISION, RECALL, F1-SCORE PADA 6 KELAS)")
    print("=" * 110)
    print(df_tabel_2.to_string(index=False))
    df_tabel_2.to_csv(os.path.join(OUTPUT_DIR, "2_tabel_metrik_per_kelas.csv"), index=False)
    simpan_tabel_ke_png(
        df_tabel_2,
        "Tabel 2. Perbandingan Precision, Recall, dan F1-Score per Kelas Penyakit Daun Cabai",
        "2_tabel_metrik_per_kelas.png",
        lebar_fig=15
    )

    # TABEL 3: TABEL CONFUSION MATRIX
    df_cm_mnet = pd.DataFrame(
        mnet["cm"],
        index=[f"Aktual: {c}" for c in class_names],
        columns=[f"Pred: {c}" for c in class_names]
    )
    df_cm_rnet = pd.DataFrame(
        rnet["cm"],
        index=[f"Aktual: {c}" for c in class_names],
        columns=[f"Pred: {c}" for c in class_names]
    )

    print("\n" + "=" * 95)
    print("TABEL 3A: TABEL CONFUSION MATRIX - MOBILENETV2")
    print("=" * 95)
    print(df_cm_mnet.to_string())
    df_cm_mnet.to_csv(os.path.join(OUTPUT_DIR, "3a_tabel_cm_mobilenetv2.csv"))

    print("\n" + "=" * 95)
    print("TABEL 3B: TABEL CONFUSION MATRIX - RESNET-50")
    print("=" * 95)
    print(df_cm_rnet.to_string())
    df_cm_rnet.to_csv(os.path.join(OUTPUT_DIR, "3b_tabel_cm_resnet50.csv"))

    # TABEL 4: RIWAYAT TRAINING PER EPOCH
    if mnet["history"] and rnet["history"]:
        jml_ep = min(len(mnet["history"]["accuracy"]), len(rnet["history"]["accuracy"]))
        data_epoch = []
        for ep in range(jml_ep):
            data_epoch.append([
                f"Epoch {ep + 1:02d}",
                f"{mnet['history']['accuracy'][ep] * 100:.2f}%",
                f"{mnet['history']['val_accuracy'][ep] * 100:.2f}%",
                f"{mnet['history']['loss'][ep]:.4f}",
                f"{mnet['history']['val_loss'][ep]:.4f}",
                f"{rnet['history']['accuracy'][ep] * 100:.2f}%",
                f"{rnet['history']['val_accuracy'][ep] * 100:.2f}%",
                f"{rnet['history']['loss'][ep]:.4f}",
                f"{rnet['history']['val_loss'][ep]:.4f}",
            ])
        df_tabel_4 = pd.DataFrame(
            data_epoch,
            columns=[
                "Epoch",
                "MNet Train Acc", "MNet Val Acc", "MNet Train Loss", "MNet Val Loss",
                "ResNet Train Acc", "ResNet Val Acc", "ResNet Train Loss", "ResNet Val Loss"
            ]
        )
        print("\n" + "=" * 115)
        print("TABEL 4: LOG PROGRES TRAINING & VALIDASI PER EPOCH (1 - 25 EPOCH)")
        print("=" * 115)
        print(df_tabel_4.to_string(index=False))
        df_tabel_4.to_csv(os.path.join(OUTPUT_DIR, "4_tabel_riwayat_25_epoch.csv"), index=False)
        simpan_tabel_ke_png(
            df_tabel_4,
            "Tabel 4. Log Progres Akurasi dan Loss per Epoch (Epoch 1 - 25)",
            "4_tabel_riwayat_25_epoch.png",
            lebar_fig=15,
            tinggi_baris=0.38
        )

    print("\n" + "=" * 95)
    print(f"[SELESAI] Semua tabel (.csv dan gambar .png) telah tersimpan di folder:\n-> {OUTPUT_DIR}")
    print("=" * 95)