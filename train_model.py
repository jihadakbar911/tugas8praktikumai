# ==============================================================================
# 1. IMPORT LIBRARY
# ==============================================================================
# Digunakan untuk operasi sistem seperti membuat folder dan navigasi path
import os
# Digunakan untuk komputasi numerik, terutama array
import numpy as np
# Digunakan untuk memuat dan memproses gambar
import cv2
# Digunakan untuk membuat plot dan visualisasi data
import matplotlib.pyplot as plt
# Digunakan untuk visualisasi data yang lebih menarik, seperti heatmap
import seaborn as sns
# Digunakan untuk membagi dataset, membuat model, dan evaluasi
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
# Digunakan untuk menyimpan dan memuat model yang sudah dilatih
import joblib

print("Semua library berhasil di-import.")
print("-" * 50)


# ==============================================================================
# 2. KONFIGURASI DAN PEMUATAN DATASET
# ==============================================================================
print("Langkah 2: Memuat Dataset dari Folder Anda...")

# --- (PENTING!) UBAH PATH DI BAWAH INI SESUAI LOKASI FOLDER DATASET ANDA ---
# Contoh untuk Windows: "C:/Users/NamaAnda/Downloads/Garbage classification"
# Contoh untuk macOS/Linux: "/home/NamaAnda/Downloads/Garbage classification"
DATASET_PATH = "C:/Users/Ojess/Downloads/archive (2)/Garbage classification/Garbage classification" # <--- EDIT BARIS INI

# Cek apakah path yang diberikan valid
if not os.path.exists(DATASET_PATH) or not os.path.isdir(DATASET_PATH):
    print("\n!!! ERROR !!!")
    print(f"Path dataset tidak ditemukan di: '{DATASET_PATH}'")
    print("Pastikan Anda sudah mengubah variabel DATASET_PATH di dalam kode.")
    exit() # Hentikan eksekusi jika path tidak valid

IMAGE_SIZE = (128, 128) # Ukuran standar untuk semua gambar

images = []
labels = []

# Mengambil nama kelas dari nama folder di dalam dataset path
class_names = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])

if not class_names:
    raise ValueError(f"Tidak ada folder kelas yang ditemukan di '{DATASET_PATH}'")

print(f"Kelas yang terdeteksi: {class_names}")

# Mengubah nama kelas (string) menjadi angka (integer)
label_encoder = LabelEncoder()
label_encoder.fit(class_names)

print("Memulai proses memuat gambar. Ini mungkin memakan waktu beberapa saat...")
for class_name in class_names:
    class_path = os.path.join(DATASET_PATH, class_name)
    image_count = 0
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, IMAGE_SIZE)
            images.append(img)
            labels.append(class_name)
            image_count += 1
        except Exception as e:
            print(f"Gagal memuat atau memproses gambar {img_path}: {e}")
    print(f"- Selesai memuat {image_count} gambar dari kelas: {class_name}")

images = np.array(images)
labels = np.array(labels)
numerical_labels = label_encoder.transform(labels)

print(f"\nJumlah total gambar yang berhasil dimuat: {len(images)}")
print(f"Bentuk (shape) array gambar: {images.shape}")
print("-" * 50)


# ==============================================================================
# 3. PREPROCESSING DATA
# ==============================================================================
print("Langkah 3: Preprocessing Data...")
n_samples = images.shape[0]
X_flattened = images.reshape(n_samples, -1)
X_normalized = X_flattened / 255.0

X = X_normalized
y = numerical_labels

print(f"Bentuk data fitur (X) setelah preprocessing: {X.shape}")
print("-" * 50)


# ==============================================================================
# 4. PISAHKAN DATA LATIH DAN UJI
# ==============================================================================
print("Langkah 4: Memisahkan Data Latih dan Uji...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(f"Ukuran data latih: {X_train.shape[0]} sampel")
print(f"Ukuran data uji: {X_test.shape[0]} sampel")
print("-" * 50)


# ==============================================================================
# 5. BUAT DAN LATIH MODEL RANDOM FOREST
# ==============================================================================
print("Langkah 5: Membuat dan Melatih Model Random Forest...")
print("Proses training sedang berjalan, mohon tunggu...")
model_rf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1, max_depth=30)
model_rf.fit(X_train, y_train)
print("Model berhasil dilatih.")
print("-" * 50)


# ==============================================================================
# 6. PREDIKSI DAN EVALUASI MODEL (LENGKAP DENGAN VISUALISASI)
# ==============================================================================
print("Langkah 6: Melakukan Prediksi dan Evaluasi Model...")
y_pred = model_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAkurasi Model: {accuracy:.4f}")

print("\nLaporan Klasifikasi:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# --- VISUALISASI 1: CONFUSION MATRIX ---
print("\nMenampilkan Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel("Prediksi Model")
plt.ylabel("Label Sebenarnya")
plt.title("Confusion Matrix")
plt.show()

# --- VISUALISASI 2: FEATURE IMPORTANCE MAP (BARU) ---
print("\nMembuat Visualisasi Peta Pentingnya Fitur (Feature Importance Map)...")
importances = model_rf.feature_importances_
importance_map = importances.reshape(IMAGE_SIZE)

plt.figure(figsize=(8, 8))
plt.title("Peta Pentingnya Fitur (Feature Importance Map)")
plt.imshow(importance_map, cmap='hot', interpolation='nearest')
plt.colorbar(label="Tingkat Kepentingan Piksel")
plt.axis('off')
plt.show()
print("-" * 50)


# ==============================================================================
# 7. SIMPAN MODEL DAN LABEL ENCODER
# ==============================================================================
print("Langkah 7: Menyimpan Model dan Label Encoder...")
os.makedirs("models", exist_ok=True)
MODEL_PATH = 'models/waste_classification_rf_model.joblib'
joblib.dump(model_rf, MODEL_PATH)

ENCODER_PATH = 'models/waste_label_encoder.joblib'
joblib.dump(label_encoder, ENCODER_PATH)

print(f"Model berhasil disimpan di: {MODEL_PATH}")
print(f"Label Encoder berhasil disimpan di: {ENCODER_PATH}")
print("\nProses selesai! File model (.joblib) sudah siap untuk digunakan di aplikasi web Flask.")
print("-" * 50)