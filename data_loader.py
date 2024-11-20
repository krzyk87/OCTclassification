from tensorflow import keras
import random
import numpy as np
import cv2
import os
from tqdm import tqdm
import skimage


class CustomGenerator(keras.utils.Sequence):
    def __init__(self, folder_path, map_characters, image_size, batch_size=1, shuffle=False, class_num=4, subset=None):
        self.file_list = self.get_file_list(folder_path, map_characters, subset)
        if shuffle:
            random.shuffle(self.file_list)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_classes = class_num
        self.list_ids = np.arange(len(self.file_list))

    def __len__(self):
        return np.ceil(len(self.file_list) / float(self.batch_size)).astype(int)

    def __getitem__(self, item):
        batch = self.file_list[item * self.batch_size:(item + 1) * self.batch_size]
        batch_x = [idx[0] for idx in batch]
        batch_y = [idx[1] for idx in batch]
        if self.image_size is not None:
            img_batch = np.array([cv2.resize(cv2.imread(file_name), (self.image_size, self.image_size))/255. for file_name in batch_x])
        else:
            img_batch = np.array([cv2.imread(file_name)/255. for file_name in batch_x])
        labels = keras.utils.to_categorical(np.array(batch_y), num_classes=self.num_classes)
        return img_batch, labels        # batch_x

    def get_file_list(self, folder_path, map_characters, subset):
        print(f'Getting list of files from {folder_path}...')
        out_list = []
        for folder_name in os.listdir(folder_path):
            if not folder_name.startswith('.'):
                label = map_characters.get(folder_name)
                if subset is not None:
                    if folder_name in subset:
                        for image_filename in tqdm(os.listdir(folder_path + folder_name)):
                            out_list.append([os.path.join(folder_path, folder_name, image_filename), label])
                else:
                    for image_filename in tqdm(os.listdir(folder_path + folder_name)):
                        out_list.append([os.path.join(folder_path, folder_name, image_filename), label])
        return out_list

    def get_labels(self):
        labels = [item[1] for item in self.file_list]
        return labels


def get_data(folder, map_characters, imageSize):
    """
    Load the data and labels from the given folder.
    """
    print(f'Loading {folder} files into memory...')
    X = []
    y = []
    for folderName in os.listdir(folder):
        if not folderName.startswith('.'):
            label = map_characters.get(folderName)
            for image_filename in tqdm(os.listdir(folder + folderName)):
                img_file = cv2.imread(folder + folderName + '/' + image_filename)
                if img_file is not None:
                    img_file = skimage.transform.resize(img_file, (imageSize, imageSize, 3))
                    img_arr = np.asarray(img_file)
                    X.append(img_arr)
                    y.append(label)
    X = np.asarray(X)
    y = np.asarray(y)
    return X, y
