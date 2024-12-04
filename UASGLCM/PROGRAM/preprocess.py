import cv2

def preprocess_image(image_path, output_path):
    # Membaca gambar
    image = cv2.imread(image_path)
    
    # Mengecek apakah gambar berhasil dibaca
    if image is None:
        print("Error: Gambar tidak dapat dibaca.")
        return
    
    # Mengubah gambar menjadi grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Mengubah ukuran gambar menjadi 960x540 pixel
    resized_image = cv2.resize(grayscale_image, (960, 540))
    
    # Menyimpan gambar hasil preprocessing
    cv2.imwrite(output_path, resized_image)
    
    # Menampilkan gambar hasil preprocessing
    cv2.imshow('Processed Image', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Contoh penggunaan
image_path = '/home/arfandiqa/VISKOM/UASGLCM/DATASET/ban fandi rusak/ban fandi rusak samping_000/ban fandi rusak samping_000.jpg'  # Ganti dengan path gambar yang akan diproses
output_path = '/home/arfandiqa/VISKOM/UASGLCM/OUTPUT/output_image.jpg'  # Ganti dengan path untuk menyimpan gambar hasil

preprocess_image(image_path, output_path)
