import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Path gambar yang akan ditampilkan
gambar_paths = [
    "/home/arfandiqa/VISKOM/UASGLCM/OUTPUT/GAMBARTESTING/baru/banaus_baru_predict.jpeg",
    "/home/arfandiqa/VISKOM/UASGLCM/OUTPUT/GAMBARTESTING/baru/banretak_baru_predict.jpeg",
    "/home/arfandiqa/VISKOM/UASGLCM/OUTPUT/GAMBARTESTING/baru/bantidakaus_baru_predict.jpeg",
    "/home/arfandiqa/VISKOM/UASGLCM/OUTPUT/GAMBARTESTING/baru/bantidakretak_baru_predict.jpeg"
]

# Keterangan untuk setiap gambar
keterangan = [
    "Ban Aus",
    "Ban Retak",
    "Ban Tidak Aus",
    "Ban Tidak Retak"
]

# Membuat figure untuk menampilkan gambar
fig, axs = plt.subplots(2, 2, figsize=(10, 10))  # 2x2 grid untuk gambar
fig.suptitle("Gambar Baru Yang DIujikan Pada Model SVM", fontsize=16)  # Judul gambar

# Menampilkan setiap gambar pada posisi yang sesuai
for i, ax in enumerate(axs.flat):
    img = mpimg.imread(gambar_paths[i])  # Membaca gambar
    ax.imshow(img)  # Menampilkan gambar
    ax.set_title(keterangan[i])  # Menambahkan keterangan di atas gambar
    ax.axis('off')  # Menonaktifkan axis

# Menyimpan hasilnya dalam file
output_path = "/home/arfandiqa/VISKOM/UASGLCM/OUTPUT/GAMBARTESTING/baru/perbandinganbaru.jpeg"
plt.savefig(output_path, bbox_inches='tight', dpi=300)

# Menampilkan gambar hasilnya
plt.show()
