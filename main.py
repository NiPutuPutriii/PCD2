import cv2
import os
import numpy as np

inputFolder = 'D:\PCDD2\Dataset_Ori'
outputFolder = 'D:\PCDD2\Dataset_Augmented'

def saturation(image, factor):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = hsv[:, :, 1] * factor
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return result

def brightness(image, factor):
    result = cv2.convertScaleAbs(image, alpha=factor, beta=0)
    return result

def sharpen(image):
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    result = cv2.filter2D(image, -1, kernel)
    return result

def blur(image, factor):
    result = cv2.GaussianBlur(image, (5, 5), factor)
    return result

# Fungsi untuk mendapatkan nama teknik berdasarkan fungsi
def get_technique_name(technique):
    if technique == saturation:
        return "Saturate"
    elif technique == brightness:
        return "Bright"
    elif technique == sharpen:
        return "Sharp"
    elif technique == blur:
        return "Blur"

# List dari fungsi augmentasi yang ingin digunakan
augmentation_functions = [saturation, brightness, sharpen, blur]

# Loop melalui semua direktori dan subdirektori di dalam inputFolder
for root, dirs, files in os.walk(inputFolder):
    # Path folder output 
    output_subfolder = os.path.join(outputFolder, os.path.relpath(root, inputFolder))

    # Loop melalui setiap file gambar dalam subfolder
    for image_file in files:
        # Baca gambar dari subfolder
        image_path = os.path.join(root, image_file)
        image = cv2.imread(image_path)

        # Loop melalui setiap fungsi augmentasi
        for augment_func in augmentation_functions:
            # Penggunaan fungsi augmentasi
            for value in [1.3, 1.5, 1.8, 2.0]:  # Variasi nilai untuk setiap teknik
                if augment_func == sharpen:
                    augmented_image = augment_func(image)
                else:
                    augmented_image = augment_func(image, value)

                # Memberikan nama teknik dan nilai
                technique_name = get_technique_name(augment_func)
                output_value_folder = os.path.join(output_subfolder, f"{technique_name}")

                # Membuat folder output jika belum ada
                os.makedirs(output_value_folder, exist_ok=True)

                # Menentukan path untuk menyimpan gambar yang sudah di-augmentasi
                output_path = os.path.join(output_value_folder, f"{value:.1f}_{image_file}")

                # Menyimpan gambar yang sudah di-augmentasi ke folder output
                cv2.imwrite(output_path, augmented_image)

print("Augmentation completed.") # Agar mengetahui kapan proses augmentasi data selesai