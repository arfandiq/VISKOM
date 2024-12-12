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

    try:
        textures = mahotas.features.texture.haralick(image)
        contrast = textures[:, 0]
        energy = textures[:, 1]
        homogeneity = textures[:, 4]

        # Konsisten dengan fitur pada program pelatihan
        features = np.hstack([
            contrast.mean(), contrast.var(), contrast.max(),
            energy.mean(), energy.var(), energy.max(),
            homogeneity.mean(), homogeneity.var(), homogeneity.max()
        ])
        return features
    except Exception as e:
        print(f"Error: Ekstraksi fitur dari {image_path} gagal. Detail: {e}")
        return None

# -----------------------------------------------------------
# Fungsi Preprocessing Gambar
# -----------------------------------------------------------

def preprocess_image(image_path, output_folder):
    """Proses gambar: konversi ke grayscale, resize, dan simpan sebagai JPEG."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"File {image_path} tidak ditemukan atau tidak bisa dibaca.")
            return None

        # Grayscale dan resize
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(grayscale_image, (960, 540))

        # Menyimpan gambar hasil preprocessing
        file_count = len([name for name in os.listdir(output_folder) if name.endswith('.jpeg')])
        filename = f"gambarbaru_{file_count + 1}.jpeg"
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, resized_image)
        print(f"Gambar telah diproses dan disimpan sebagai {filename}")
        return output_path
    except Exception as e:
        print(f"Error dalam preprocessing gambar: {e}")
        return None

# -----------------------------------------------------------
# Fungsi untuk Prediksi Gambar Baru dengan Model SVM
# -----------------------------------------------------------

def predict_damage(image_path, model, output_folder):
    # Preprocessing gambar
    preprocessed_image_path = preprocess_image(image_path, output_folder)
    if preprocessed_image_path is None:
        return

    # Ekstraksi fitur GLCM
    feature_vector = extract_glcm_features(preprocessed_image_path)
    if feature_vector is None:
        return

    # Prediksi menggunakan model SVM
    prediction = model.predict([feature_vector])[0]

    # Menentukan teks hasil prediksi
    class_labels = {
        1: "Kerusakan: Ban Rusak (Keretakan Samping)",
        2: "Kerusakan: Ban Rusak (Aus)",
        3: "Kerusakan: Ban Bagus (Tidak Retak)",
        4: "Kerusakan: Ban Bagus (Tidak Aus)"
    }
    result_text = class_labels.get(prediction, "Kerusakan: Tidak Diketahui")
    print(f"Hasil Prediksi: {result_text}")

    # Membaca ulang gambar yang telah diproses untuk menambahkan teks hasil prediksi
    resized_image = cv2.imread(preprocessed_image_path)

    # Menambahkan teks ke gambar
    cv2.putText(resized_image, result_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Menyimpan gambar hasil prediksi
    predicted_filename = f"predicted_{os.path.basename(preprocessed_image_path)}"
    predicted_path = os.path.join(output_folder, predicted_filename)
    cv2.imwrite(predicted_path, resized_image)
    print(f"Hasil prediksi disimpan di {predicted_path}")

# -----------------------------------------------------------
# Main Program
# -----------------------------------------------------------

if __name__ == '__main__':
    # Path ke model SVM yang sudah dilatih
    model_path = '/home/arfandiqa/VISKOM/UASGLCM/OUTPUT/modelsvm/model_svm_terbaru_banget.pkl'

    # Memuat model SVM
    model = joblib.load(model_path)
    print(f"Model SVM dimuat dari {model_path}")

    # Path gambar input untuk prediksi
    image_path = '/home/arfandiqa/VISKOM/FILEPENTING/gambartesting/bantidakretak_dataset.jpg'  # Ganti sesuai file input

    # Folder untuk menyimpan hasil prediksi
    output_folder = '/home/arfandiqa/VISKOM/UASGLCM/OUTPUT/hasilprogram'
    os.makedirs(output_folder, exist_ok=True)

    # Memproses gambar untuk prediksi
    predict_damage(image_path, model, output_folder)
