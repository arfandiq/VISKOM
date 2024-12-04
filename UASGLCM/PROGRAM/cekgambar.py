import os

# Folder input gambar
input_ban_retak = '/home/arfandiqa/VISKOM/UASGLCM/DATASET/banretak'
input_ban_aus = '/home/arfandiqa/VISKOM/UASGLCM/DATASET/banaus'
input_ban_tidakretak = '/home/arfandiqa/VISKOM/UASGLCM/DATASET/bantidakretak'
input_ban_tidakaus = '/home/arfandiqa/VISKOM/UASGLCM/DATASET/bantidakaus'

# Daftar folder yang akan diperiksa
folders = [input_ban_retak, input_ban_aus, input_ban_tidakretak, input_ban_tidakaus]

# Format gambar yang akan dianggap valid
valid_image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

for folder in folders:
    if not os.path.exists(folder):
        print(f"Folder {folder} tidak ditemukan.")
    else:
        images = os.listdir(folder)
        
        # Menyaring hanya file gambar berdasarkan ekstensi
        image_files = [img for img in images if any(img.lower().endswith(ext) for ext in valid_image_extensions)]
        
        if len(image_files) == 0:
            print(f"Folder {folder} kosong atau tidak berisi gambar.")
        else:
            print(f"Folder {folder} berisi {len(image_files)} gambar.")
