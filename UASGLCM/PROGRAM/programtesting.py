import os
import cv2
import numpy as np
import mahotas
import joblib

# -----------------------------------------------------------
# Fungsi untuk Ekstraksi Fitur GLCM dengan Mahotas
# -----------------------------------------------------------

def extract_glcm_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"File {image_path} tidak bisa dibaca.")
        return None

    textures = mahotas.features.texture.haralick(image)
    contrast = textures[:, 0]
    energy = textures[:, 1]
    homogeneity = textures[:, 4]

    features = np.hstack([contrast.mean(), energy.mean(), homogeneity.mean()])
    return features

# -----------------------------------------------------------
# Fungsi untuk Prediksi Kerusakan Gambar Baru dengan Model SVM
# -----------------------------------------------------------

def preprocess_image(image_path, output_folder):
    """Proses gambar: konversi ke grayscale, resize, dan simpan sebagai JPEG dengan nama yang berurutan."""
    # Membaca gambar
    image = cv2.imread(image_path)

    # Mengonversi gambar ke grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize gambar
    resized_image = cv2.resize(grayscale_image, (960, 540))

    # Menyimpan gambar yang sudah diproses dalam format JPEG dengan nama yang berurutan
    file_count = len([name for name in os.listdir(output_folder) if name.endswith('.jpeg')])
    filename = f"gambarbaru_{file_count + 1}.jpeg"
    output_path = os.path.join(output_folder, filename)

    # Menyimpan gambar yang telah diproses
    cv2.imwrite(output_path, resized_image)
    print(f"Gambar telah diproses dan disimpan sebagai {filename}")
    
    return output_path

def predict_damage(image_path, model, output_folder):
    # Preprocess gambar terlebih dahulu
    preprocessed_image_path = preprocess_image(image_path, output_folder)

    # Mengekstrak fitur GLCM dari gambar yang sudah diproses
    feature_vector = extract_glcm_features(preprocessed_image_path)
    if feature_vector is None:
        return

    # Prediksi menggunakan model SVM
    prediction = model.predict([feature_vector])

    # Menentukan teks hasil prediksi
    if prediction == 1:
        result_text = "Kerusakan: Ban Rusak (Keretakan Samping)"
    elif prediction == 2:
        result_text = "Kerusakan: Ban Rusak (Aus)"
    else:
        result_text = "Kerusakan: Ban Bagus (Tidak Retak / Tidak Aus)"
    
    # Membaca gambar yang sudah diproses untuk menambahkan teks hasil prediksi
    resized_image = cv2.imread(preprocessed_image_path)
    
    # Menambahkan teks pada gambar hasil prediksi
    cv2.putText(resized_image, result_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Menyimpan gambar hasil prediksi ke folder output
    filename = os.path.basename(preprocessed_image_path)
    output_path = os.path.join(output_folder, f"predicted_{filename}")
    
    # Menyimpan gambar dengan hasil prediksi
    cv2.imwrite(output_path, resized_image)
    print(f"Hasil prediksi disimpan di {output_path}")

# -----------------------------------------------------------
# Main Program
# -----------------------------------------------------------

if __name__ == '__main__':
    # Path ke model SVM yang sudah disimpan
    model_path = '/home/arfandiqa/VISKOM/UASGLCM/OUTPUT/modelsvm/model_svm_terbaru_banget.pkl'

    # Memuat model SVM
    model = joblib.load(model_path)
    print(f"Model SVM dimuat dari {model_path}")

    # Gambar input yang ingin diuji (gunakan path gambar tunggal)
    image_path = '/home/arfandiqa/VISKOM/UASGLCM/DATASET/gambarbaru/jetlagi.jpeg'  # Ganti dengan path gambar tunggal yang ingin diuji

    # Folder untuk menyimpan hasil prediksi
    output_folder = '/home/arfandiqa/VISKOM/UASGLCM/OUTPUT/hasilprogram'

    # Membuat folder hasil prediksi jika belum ada
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Memproses gambar tunggal untuk prediksi
    print(f"Memproses gambar: {image_path}")
    predict_damage(image_path, model, output_folder)
