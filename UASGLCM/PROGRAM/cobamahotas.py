import os
import cv2
import numpy as np
import mahotas
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------
# Fungsi untuk Preprocessing Gambar: Grayscale dan Resize
# -----------------------------------------------------------

def preprocess_image(input_path, output_path, target_size=(960, 540)):
    image = cv2.imread(input_path)
    if image is None:
        print(f"File {input_path} tidak ditemukan atau tidak bisa dibaca.")
        return False

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, target_size)
    cv2.imwrite(output_path, resized_image)
    print(f"Proses preprocessing gambar berhasil: {output_path}")
    return True

def preprocess_dataset(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    folder_name = os.path.basename(input_folder)
    file_counter = 1
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_filename = f"{folder_name}_{file_counter}.jpeg"
            output_path = os.path.join(output_folder, output_filename)
            if not preprocess_image(input_path, output_path):
                print(f"Skipping file {filename} due to error during preprocessing.")
            file_counter += 1

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

def extract_features_from_dataset(output_folder):
    features = []
    labels = []

    for filename in os.listdir(output_folder):
        image_path = os.path.join(output_folder, filename)
        feature_vector = extract_glcm_features(image_path)
        if feature_vector is None:
            continue
        if 'banretak' in filename:
            features.append(feature_vector)
            labels.append(1)
        elif 'banaus' in filename:
            features.append(feature_vector)
            labels.append(2)
        elif 'bantidakretak' in filename:
            features.append(feature_vector)
            labels.append(3)
        elif 'bantidakaus' in filename:
            features.append(feature_vector)
            labels.append(4)

    if len(features) == 0 or len(labels) == 0:
        print("Tidak ada fitur atau label yang diekstraksi. Pastikan folder input berisi gambar yang benar.")
        return None, None

    return np.array(features), np.array(labels)

# -----------------------------------------------------------
# Pelatihan Model SVM
# -----------------------------------------------------------

def train_svm(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    # Evaluasi Model
    print("Evaluasi Model SVM:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # Simpan Confusion Matrix sebagai Gambar
    cm_image_path = '/home/arfandiqa/VISKOM/UASGLCM/OUTPUT/modelsvm/confusion_matrix_terbaru.png'
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=['Ban Rusak (Keretakan Samping)', 'Ban Rusak (Aus)', 'Ban Bagus (Tidak Retak)', 'Ban Bagus (Tidak Aus)'], 
        yticklabels=['Ban Rusak (Keretakan Samping)', 'Ban Rusak (Aus)', 'Ban Bagus (Tidak Retak)', 'Ban Bagus (Tidak Aus)']
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Prediksi")
    plt.ylabel("Sebenarnya")
    plt.savefig(cm_image_path)
    plt.close()
    
    # Simpan Model SVM
    model_path = '/home/arfandiqa/VISKOM/UASGLCM/OUTPUT/modelsvm/model_svm_terbaru.pkl'
    joblib.dump(clf, model_path)
    print(f"Model SVM disimpan di {model_path}")

    return clf

# -----------------------------------------------------------
# Fungsi Prediksi Gambar Baru dan Menampilkan Gambar
# -----------------------------------------------------------

def predict_damage(image_path, model):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(image, (960, 540))
    feature_vector = extract_glcm_features(image_path)
    prediction = model.predict([feature_vector])
    
    if prediction == 1:
        print(f"Gambar {image_path} menunjukkan kerusakan samping (keretakan).")
        result_text = "Kerusakan: Ban Rusak (Keretakan Samping)"
    elif prediction == 2:
        print(f"Gambar {image_path} menunjukkan ban aus.")
        result_text = "Kerusakan: Ban Rusak (Aus)"
    elif prediction == 3:
        print(f"Gambar {image_path} menunjukkan ban aus.")
        result_text = "Kerusakan: Ban Bagus (Tidak Retak)"
    elif prediction == 4:
        print(f"Gambar {image_path} menunjukkan ban aus.")
        result_text = "Kerusakan: Ban Bagus (Tidak Aus)"
    else:
        print(f"Gambar {image_path} menunjukkan ban tidak diketahui.")
        result_text = "Kerusakan: ban tidak diketahui"
    
    cv2.putText(resized_image, result_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Hasil Prediksi", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# -----------------------------------------------------------
# Main Program
# -----------------------------------------------------------

if __name__ == '__main__':
    input_ban_retak = '/home/arfandiqa/VISKOM/UASGLCM/DATASET/banretak'
    input_ban_aus = '/home/arfandiqa/VISKOM/UASGLCM/DATASET/banaus'
    input_ban_tidakretak = '/home/arfandiqa/VISKOM/UASGLCM/DATASET/bantidakretak'
    input_ban_tidakaus = '/home/arfandiqa/VISKOM/UASGLCM/DATASET/bantidakaus'

    output_ban_retak = '/home/arfandiqa/VISKOM/UASGLCM/OUTPUT/banretak_preproc'
    output_ban_aus = '/home/arfandiqa/VISKOM/UASGLCM/OUTPUT/banaus_preproc'
    output_ban_tidakretak = '/home/arfandiqa/VISKOM/UASGLCM/OUTPUT/bantidakretak_preproc'
    output_ban_tidakaus = '/home/arfandiqa/VISKOM/UASGLCM/OUTPUT/bantidakaus_preproc'

    print("Preprocessing gambar ban rusak (keretakan samping)...")
    preprocess_dataset(input_ban_retak, output_ban_retak)

    print("Preprocessing gambar ban aus...")
    preprocess_dataset(input_ban_aus, output_ban_aus)

    print("Preprocessing gambar ban bagus (tidak retak)...")
    preprocess_dataset(input_ban_tidakretak, output_ban_tidakretak)

    print("Preprocessing gambar ban tidak aus...")
    preprocess_dataset(input_ban_tidakaus, output_ban_tidakaus)

    features_retak, labels_retak = extract_features_from_dataset(output_ban_retak)
    features_aus, labels_aus = extract_features_from_dataset(output_ban_aus)
    features_bagus, labels_bagus = extract_features_from_dataset(output_ban_tidakretak)
    features_tidakaus, labels_tidakaus = extract_features_from_dataset(output_ban_tidakaus)

    if features_retak is None:
        print("Tidak ada fitur yang berhasil diekstraksi dari ban retak.")
    else:
        features = np.concatenate([features_retak, features_aus, features_bagus, features_tidakaus])
        labels = np.concatenate([labels_retak, labels_aus, labels_bagus, labels_tidakaus])

        print("Melatih model SVM...")
        model = train_svm(features, labels)

        image_to_predict = '/home/arfandiqa/VISKOM/UASGLCM/OUTPUT/bantidakaus_preproc/bantidakaus_144.jpeg'
        predict_damage(image_to_predict, model)
