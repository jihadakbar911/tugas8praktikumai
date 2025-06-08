# ==============================================================================
# IMPORT LIBRARY
# ==============================================================================
import os
import cv2
import numpy as np
import joblib
from flask import Flask, render_template, request, url_for

import logging

logging.basicConfig(level=logging.INFO)

# ==============================================================================
# INISIALISASI APLIKASI FLASK DAN KONFIGURASI
# ==============================================================================
# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Konfigurasi folder untuk menyimpan file yang diunggah
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Ukuran gambar harus SAMA PERSIS dengan saat training
IMAGE_SIZE = (128, 128)

# ==============================================================================
# MUAT MODEL DAN ENCODER YANG SUDAH DILATIH
# ==============================================================================
# Muat model Random Forest dan Label Encoder yang sudah kita simpan sebelumnya
# Model ini dimuat sekali saja saat aplikasi pertama kali dijalankan
try:
    model = joblib.load('models/waste_classification_rf_model.joblib')
    label_encoder = joblib.load('models/waste_label_encoder.joblib')
    print("Model dan Label Encoder berhasil dimuat.")
except FileNotFoundError:
    print("!!! ERROR: File model atau encoder tidak ditemukan. Pastikan folder 'models' ada di direktori yang sama dengan app.py")
    model = None
    label_encoder = None

# ==============================================================================
# FUNGSI UNTUK PREPROCESSING GAMBAR
# ==============================================================================
def preprocess_image(image_path):
    """
    Fungsi untuk melakukan preprocessing pada gambar yang diunggah pengguna.
    Langkah-langkahnya harus sama persis dengan saat training.
    """
    try:
        # Baca gambar
        img = cv2.imread(image_path)
        # Ubah ke Grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Ubah ukuran gambar
        img = cv2.resize(img, IMAGE_SIZE)
        # "Meratakan" gambar menjadi vektor 1D
        img_flattened = img.reshape(1, -1)
        # Normalisasi nilai piksel
        img_normalized = img_flattened / 255.0
        return img_normalized
    except Exception as e:
        print(f"Error saat memproses gambar: {e}")
        return None

# ==============================================================================
# ROUTING APLIKASI WEB
# ==============================================================================
@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Fungsi untuk menangani halaman utama (GET) dan proses unggah gambar (POST).
    """
    if request.method == 'POST':
        # Cek apakah ada file yang diunggah
        if 'file' not in request.files:
            return render_template('index.html', error="Tidak ada file yang dipilih.")
        
        file = request.files['file']
        
        # Cek jika pengguna tidak memilih file
        if file.filename == '':
            return render_template('index.html', error="Tidak ada file yang dipilih.")
            
        if file and model:
            # Simpan file yang diunggah ke folder 'static/uploads'
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Lakukan preprocessing pada gambar yang baru diunggah 
            processed_image = preprocess_image(filepath)
            
            if processed_image is not None:
                # Lakukan prediksi menggunakan model Random Forest 
                prediction_numeric = model.predict(processed_image)
                # Terjemahkan hasil numerik menjadi label string (e.g., 'glass')
                prediction_label = label_encoder.inverse_transform(prediction_numeric)[0]
                
                # Tampilkan hasil klasifikasi pada halaman web 
                return render_template('result.html', prediction=prediction_label.capitalize(), image_file=filename)
            else:
                return render_template('index.html', error="Gagal memproses gambar.")

    # Tampilkan halaman utama untuk mengunggah gambar 
    return render_template('index.html')


# ==============================================================================
# MENJALANKAN APLIKASI
# ==============================================================================
if __name__ == '__main__':
    # Buat folder 'uploads' jika belum ada
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    # Jalankan aplikasi Flask 
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
