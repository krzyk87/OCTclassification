import cv2
import os
from tqdm import tqdm
import time

subset_list = {"train", "val", "test"}
disease_list = {"CNV", "DME", "DRUSEN", "NORMAL"}
time_total = 0
no_all = 0

equalization_method = 'CLAHE'

if 'CLAHE' in equalization_method:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

for subset in subset_list:
    for disease in disease_list:

        input_folder = "../../DATA/OCT2017/" + subset + "/" + disease
        output_folder = "../../DATA/OCT2017_" + equalization_method + "/" + subset + "/" + disease

        os.makedirs(output_folder, exist_ok=True)
        no_all += len(os.listdir(input_folder))

        for file in tqdm(os.listdir(input_folder)):
            filename = os.fsdecode(file)
            dot = filename.rfind('.')

            # output_file_path = os.path.join(output_folder, filename.replace(f"_{equalization_method}".lower(), ""))
            # if not os.path.exists(output_file_path):
            #     input_file_path = os.path.join(input_folder, filename)
            #     print(f'Removing file {input_file_path}')
            #     os.remove(input_file_path)

            if not os.path.exists(os.path.join(output_folder, filename[:dot] + "_" + equalization_method.lower() + ".jpeg")):
                # load the input image from disk and convert it to grayscale
                print(f"[INFO] loading image {filename}")
                image = cv2.imread(os.path.join(input_folder, filename))
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # start = time.time()
                if 'CLAHE' in equalization_method:
                    equalized = clahe.apply(gray)
                else:
                    equalized = cv2.equalizeHist(gray)
                # end = time.time()
                # time_total += (end - start)

                # save image
                cv2.imwrite(os.path.join(output_folder, filename[:dot] + "_" + equalization_method.lower() + ".jpeg"), equalized)

print(f'Elapsed time: {time_total}')
print(f'Mean time: {time_total/no_all}')
