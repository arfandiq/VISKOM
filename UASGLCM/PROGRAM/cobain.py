import os
import cv2
import numpy as np
from skimage.feature import greycomatrix, greycoprops
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# -----------------------------------------------------------
# Fungsi untuk Preprocessing Gambar: Grayscale dan Resize
# -----------------------------------------------------------

def preprocess_image(input_path, output_path, target_size=(960, 540)):
    """
    Mengubah gambar menjadi grayscale dan resize ke ukuran target.
    """
    # Membaca gambar
    image = cv2.imread(input_path)
    if image is None:
        print(f"File {input_path} tidak ditemukan atau tidak bisa dibaca.")
        return

    # Mengubah gambar ke grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Mengubah ukuran gambar
    resized_image = cv2.resize(gray_image, target_size)

    # Menyimpan gambar yang sudah diproses
    cv2.imwrite(output_path, resized_image)
    print(f"Proses preprocessing gambar berhasil: {output_path}")

def preprocess_dataset(input_folder, output_folder):
    """
    Memproses semua gambar dalam folder dataset dan menyimpannya di folder output.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        preprocess_image(input_path, output_path)

# -----------------------------------------------------------
# Fungsi untuk Ekstraksi Fitur GLCM
# -----------------------------------------------------------

def extract_glcm_features(image_path):
    """
    Mengekstrak fitur GLCM: Kontras, Energi, dan Homogenitas
    """
    # Membaca gambar yang sudah diproses (grayscale)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Menghitung GLCM dengan jarak 1 piksel dan berbagai arah (0°, 45°, 90°, 135°)
    glcm = greycomatrix(image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)
    
    # Menghitung fitur-fitur GLCM: Kontras, Energi, Homogenitas
    contrast = greycoprops(glcm, 'contrast').flatten()
    energy = greycoprops(glcm, 'energy').flatten()
    homogeneity = greycoprops(glcm, 'homogeneity').flatten()
    
    # Menggabungkan fitur-fitur tersebut menjadi satu vektor
    features = np.hstack([contrast, energy, homogeneity])
    return features

def extract_features_from_dataset(input_folder):
    """
    Mengekstrak fitur GLCM dari semua gambar dalam folder dataset.
    """
    features = []
    labels = []
    
    # Menambahkan label untuk data kerusakan (ban rusak retak samping, ban rusak aus) dan ban bagus
    for filename in os.listdir(input_folder):
        image_path = os.path.join(input_folder, filename)
        
        # Ekstraksi fitur GLCM
        feature_vector = extract_glcm_features(image_path)
        
        # Menyimpan fitur dan label
        if 'banretak' in filename:
            features.append(feature_vector)
            labels.append(1)  # Label 1 untuk keretakan samping
        elif 'banaus' in filename:
            features.append(feature_vector)
            labels.append(2)  # Label 2 untuk ban aus
        elif 'bantidakretak' in filename:
            features.append(feature_vector)
            labels.append(0)  # Label 0 untuk ban bagus (tidak retak)
        elif 'bantidakaus' in filename:
            features.append(feature_vector)
            labels.append(0)  # Label 0 untuk ban bagus (tidak aus)
    
    return np.array(features), np.array(labels)

# -----------------------------------------------------------
# Pelatihan Model SVM
# -----------------------------------------------------------

def train_svm(features, labels):
    """
    Melatih model SVM dengan data fitur dan label yang diberikan.
    """
    # Membagi dataset menjadi data pelatihan dan data pengujian
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Membuat dan melatih model SVM dengan kernel linear
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)

    # Menguji model
    y_pred = clf.predict(X_test)

    # Menampilkan laporan evaluasi
    print("Evaluasi Model SVM:")
    print(classification_report(y_test, y_pred))

    return clf

# -----------------------------------------------------------
# Fungsi Prediksi Gambar Baru dan Menampilkan Gambar
# -----------------------------------------------------------

def predict_damage(image_path, model):
    """
    Memprediksi apakah gambar baru menunjukkan kerusakan samping atau ban aus,
    dan menampilkan gambar beserta keterangan.
    """
    # Preprocessing gambar baru
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(image, (960, 540))
    
    # Ekstraksi fitur GLCM
    feature_vector = extract_glcm_features(image_path)
    
    # Prediksi dengan model SVM
    prediction = model.predict([feature_vector])
    
    # Menampilkan hasil prediksi
    if prediction == 1:
        print(f"Gambar {image_path} menunjukkan kerusakan samping (keretakan).")
        result_text = "Kerusakan: Ban Rusak (Keretakan Samping)"
    elif prediction == 2:
        print(f"Gambar {image_path} menunjukkan ban aus.")
        result_text = "Kerusakan: Ban Rusak (Aus)"
    else:
        print(f"Gambar {image_path} menunjukkan ban bagus.")
        result_text = "Kerusakan: Ban Bagus (Tidak Retak / Tidak Aus)"
    
    # Menampilkan gambar dengan keterangannya
    cv2.putText(resized_image, result_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Hasil Prediksi", resized_image)
    cv2.waitKey(0)  # Menunggu input key untuk menutup jendela
    cv2.destroyAllWindows()

# -----------------------------------------------------------
# Main Program
# -----------------------------------------------------------

if __name__ == '__main__':
    # Path ke folder gambar
    input_ban_retak = '/home/arfandiqa/VISKOM/UASGLCM/DATASET/banretak'
    input_ban_aus = '/home/arfandiqa/VISKOM/UASGLCM/DATASET/banaus'
    input_ban_tidakretak = '/home/arfandiqa/VISKOM/UASGLCM/DATASET/bantidakretak'
    input_ban_tidakaus = '/home/arfandiqa/VISKOM/UASGLCM/DATASET/bantidakaus'

    # Path ke folder output
    output_ban_retak = '/home/arfandiqa/VISKOM/UASGLCM/OUTPUT/banretak_preproc'
    output_ban_aus = '/home/arfandiqa/VISKOM/UASGLCM/OUTPUT/banaus_preproc'
    output_ban_tidakretak = '/home/arfandiqa/VISKOM/UASGLCM/OUTPUT/bantidakretak_preproc'
    output_ban_tidakaus = '/home/arfandiqa/VISKOM/UASGLCM/OUTPUT/bantidakaus_preproc'

    # Proses gambar ban rusak (keretakan samping) dan ban aus
    print("Preprocessing gambar ban rusak (keretakan samping)...")
    preprocess_dataset(input_ban_retak, output_ban_retak)

    print("Preprocessing gambar ban aus...")
    preprocess_dataset(input_ban_aus, output_ban_aus)

    # Proses gambar ban bagus (tidak retak dan tidak aus)
    print("Preprocessing gambar ban tidak retak...")
    preprocess_dataset(input_ban_tidakretak, output_ban_tidakretak)

    print("Preprocessing gambar ban tidak aus...")
    preprocess_dataset(input_ban_tidakaus, output_ban_tidakaus)

    # Ekstraksi fitur dari dataset ban rusak dan ban bagus
    print("Ekstraksi fitur GLCM dari dataset...")
    features, labels = extract_features_from_dataset(output_ban_retak)
    features_aus, labels_aus = extract_features_from_dataset(output_ban_aus)
    features_bagus, labels_bagus = extract_features_from_dataset(output_ban_tidakretak)
    features_bagus_2, labels_bagus_2 = extract_features_from_dataset(output_ban_tidakaus)

    # Gabungkan fitur dan label dari semua dataset
    all_features = np.vstack([features, features_aus, features_bagus, features_bagus_2])
    all_labels = np.hstack([labels, labels_aus, labels_bagus, labels_bagus_2])

    # Melatih model SVM
    model = train_svm(all_features, all_labels)

    # Path ke gambar yang ingin diprediksi
    image_path = '/home/arfandiqa/VISKOM/UASGLCM/DATASET/banretak/ban fandi rusak samping_000/ban fandi rusak samping_000.jpg'  # Ganti dengan path gambar yang ingin diprediksi
    predict_damage(image_path, model)
