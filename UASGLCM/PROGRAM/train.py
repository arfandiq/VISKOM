import os
import cv2
import numpy as np
import mahotas
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
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
        print(f"Error: Gambar di {image_path} tidak dapat dibaca. Periksa apakah file rusak atau format tidak didukung.")
        return None

    try:
        textures = mahotas.features.texture.haralick(image)
        contrast = textures[:, 0]
        energy = textures[:, 1]
        homogeneity = textures[:, 4]

        # Tambahkan lebih banyak statistik
        features = np.hstack([
            contrast.mean(), contrast.var(), contrast.max(),
            energy.mean(), energy.var(), energy.max(),
            homogeneity.mean(), homogeneity.var(), homogeneity.max()
        ])
        return features
    except Exception as e:
        print(f"Error: Ekstraksi fitur dari {image_path} gagal. Detail: {e}")
        return None

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
# Fungsi Balancing Dataset
# -----------------------------------------------------------

def balance_dataset(features, labels):
    print("Melakukan balancing data dengan SMOTE...")
    smote = SMOTE(random_state=42)
    features_balanced, labels_balanced = smote.fit_resample(features, labels)
    print(f"Data setelah balancing: {len(labels_balanced)} sampel.")
    return features_balanced, labels_balanced

# -----------------------------------------------------------
# Fungsi untuk Hyperparameter Tuning
# -----------------------------------------------------------

def tune_svm(X_train, y_train):
    print("Melakukan hyperparameter tuning pada SVM...")
    parameters = {'kernel': ['linear', 'rbf'], 'C': [0.1, 1, 10, 100]}
    clf = GridSearchCV(svm.SVC(), parameters, cv=5)
    clf.fit(X_train, y_train)
    print(f"Parameter terbaik: {clf.best_params_}")
    return clf.best_estimator_

# -----------------------------------------------------------
# Pelatihan Model SVM
# -----------------------------------------------------------

def train_svm(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    clf = tune_svm(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    # Evaluasi Model
    print("Evaluasi Model SVM:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_image_path = '/home/arfandiqa/VISKOM/UASGLCM/OUTPUT/modelsvm/confusion_matrix_fix_banget.png'
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Keretakan Samping', 'Aus', 'Tidak Retak', 'Tidak Aus'],
        yticklabels=['Keretakan Samping', 'Aus', 'Tidak Retak', 'Tidak Aus']
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Prediksi")
    plt.ylabel("Sebenarnya")
    plt.savefig(cm_image_path)  # Menyimpan confusion matrix ke file
    print(f"Confusion matrix disimpan di {cm_image_path}")
    plt.close()
    
    # Simpan Model SVM
    model_path = '/home/arfandiqa/VISKOM/UASGLCM/OUTPUT/modelsvm/model_svm_terbaru_banget.pkl'
    joblib.dump(clf, model_path)
    print(f"Model SVM disimpan di {model_path}")

    return clf

# -----------------------------------------------------------
# Fungsi Prediksi Gambar Baru dan Menampilkan Gambar
# -----------------------------------------------------------

def predict_damage(image_path, model):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Gambar {image_path} tidak dapat dibaca.")
        return

    try:
        feature_vector = extract_glcm_features(image_path)
        if feature_vector is None:
            print(f"Error: Fitur tidak dapat diekstraksi dari {image_path}.")
            return

        prediction = model.predict([feature_vector])[0]
        class_labels = {
            1: "Ban Rusak (Keretakan Samping)",
            2: "Ban Rusak (Aus)",
            3: "Ban Bagus (Tidak Retak)",
            4: "Ban Bagus (Tidak Aus)"
        }
        result_text = f"Kerusakan: {class_labels.get(prediction, 'Tidak Diketahui')}"
        print(result_text)

        resized_image = cv2.resize(image, (960, 540))
        cv2.putText(resized_image, result_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Hasil Prediksi", resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error: Prediksi gagal untuk {image_path}. Detail: {e}")

# -----------------------------------------------------------
# Main Program
# -----------------------------------------------------------

if __name__ == '__main__':
    input_folders = {
        'banretak': '/home/arfandiqa/VISKOM/UASGLCM/DATASET/banretak',
        'banaus': '/home/arfandiqa/VISKOM/UASGLCM/DATASET/banaus',
        'bantidakretak': '/home/arfandiqa/VISKOM/UASGLCM/DATASET/bantidakretak',
        'bantidakaus': '/home/arfandiqa/VISKOM/UASGLCM/DATASET/bantidakaus'
    }
    output_folders = {
        key: f"/home/arfandiqa/VISKOM/UASGLCM/OUTPUT/{key}_preproc" for key in input_folders
    }

    for key, input_folder in input_folders.items():
        print(f"Preprocessing gambar untuk {key}...")
        preprocess_dataset(input_folder, output_folders[key])

    features, labels = [], []
    for key, output_folder in output_folders.items():
        f, l = extract_features_from_dataset(output_folder)
        if f is not None and l is not None:
            features.append(f)
            labels.append(l)

    features = np.concatenate(features)
    labels = np.concatenate(labels)
    features, labels = balance_dataset(features, labels)

    print("Melatih model SVM...")
    model = train_svm(features, labels)

    image_to_predict = '/home/arfandiqa/VISKOM/UASGLCM/OUTPUT/bantidakaus_preproc/bantidakaus_144.jpeg'
    predict_damage(image_to_predict, model)
