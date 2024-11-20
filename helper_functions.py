# conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
# pip install numba
# ----------------------- to free gpu memory:
# from numba import cuda
# device = cuda.get_current_device()
# device.reset()
import os
import matplotlib.pyplot as plt
import random
import cv2
import numpy as np
import itertools
from tensorflow import keras
from glob import glob
import pandas as pd
from tqdm import tqdm


def count_white_edge(input_image):
    top = np.sum(input_image[0, 0, :, :] == 1) / 3
    left = np.sum(input_image[0, :, 0, :] == 1) / 3
    right = np.sum(input_image[0, :, -1, :] == 1) / 3
    bottom = np.sum(input_image[0, -1, :, :] == 1) / 3
    return left, top, right, bottom


def crop_img(input_img):
    _, rows, cols, chs = input_img.shape
    cut_img = input_img

    # check for a whole white row/column and remove it
    rows_to_remove, cols_to_remove = [], []
    row_id, col_id = 0, 0
    while row_id < rows:
        if np.min(cut_img[0, row_id, :, :]) >= 0.90:
            rows_to_remove.append(row_id)
        row_id = row_id + 1
    cut_img = np.delete(cut_img, rows_to_remove, 1)
    while col_id < cols:
        if np.min(cut_img[0, :, col_id, :]) >= 0.90:
            cols_to_remove.append(col_id)
        col_id = col_id + 1
    cut_img = np.delete(cut_img, cols_to_remove, 2)

    left_px, top_px, right_px, bottom_px = count_white_edge(cut_img)

    # find the edge with most white pixels and remove single line,
    # repeat until no white edges
    while np.any(np.nonzero([left_px, top_px, right_px, bottom_px])):
        idx = np.argmax([left_px, top_px, right_px, bottom_px])
        if idx == 0:
            cut_img = cut_image(cut_img, 'left')
            left_px, top_px, right_px, bottom_px = count_white_edge(cut_img)
        elif idx == 1:
            cut_img = cut_image(cut_img, 'top')
            left_px, top_px, right_px, bottom_px = count_white_edge(cut_img)
        elif idx == 2:
            cut_img = cut_image(cut_img, 'right')
            left_px, top_px, right_px, bottom_px = count_white_edge(cut_img)
        elif idx == 3:
            cut_img = cut_image(cut_img, 'bottom')
            left_px, top_px, right_px, bottom_px = count_white_edge(cut_img)

    # plt.subplot(1, 2, 1)
    # plt.imshow(input_img[0])
    # plt.subplot(1, 2, 2)
    # plt.imshow(cut_img[0])
    # plt.show()
    # print(f'Output image shape: {cut_img.shape}')
    return cut_img


def cut_image(img, direction):
    if direction == 'top':
        img = img[:, 1:, :, :]  # cut the left column
    elif direction == 'left':
        img = img[:, :, 1:, :]
    elif direction == 'right':
        img = img[:, :, :-2, :]
    elif direction == 'bottom':
        img = img[:, :-2, :, :]
    else:
        print('Unknown image cropping direction!')
    return img


def remove_white_edge(subset_gen, sel_class_idx):
    for i in tqdm(range(len(subset_gen))):
        img, label, img_path = subset_gen[i]
        if label[0][sel_class_idx]:
            img_path_nowhite = img_path[0].replace("OCT5class", "OCT5class_nowhite")
            output_path = os.path.dirname(img_path_nowhite)
            os.makedirs(output_path, exist_ok=True)
            if not os.path.exists(img_path_nowhite):
                if find_white_edge(img):
                    print(f'Cropping image: {img_path}')
                    img = crop_img(img)
                cv2.imwrite(img_path_nowhite, img[0]*255)


def calc_hist_data(subset_gen, hist_data, sel_class_idx):
    for i in range(len(subset_gen)):
        img, label, img_path = subset_gen[i]
        if label[0][sel_class_idx]:
            print(f'Loading image {i}...')
            hist_temp = np.histogram(np.squeeze(img)[:, :, 0], bins=256)[0]
            hist_data = np.add(hist_data, hist_temp)
    return hist_data


def find_white_edge(img):
    img = img.squeeze()
    img = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2GRAY)
    # check top row
    white_edge = False
    if np.any(img[0, :] == 1):
        white_edge = True
    # check bottom row
    if np.any(img[-1, :] == 1):
        white_edge = True
    # check left edge
    if np.any(img[:, 0] == 1):
        white_edge = True
    # check right edge
    if np.any(img[:, -1] == 1):
        white_edge = True
    return white_edge


def count_white_edge_in_subset(subset, sel_class_idx):
    num_edges_found = 0
    for i in tqdm(range(len(subset))):
        img, label = subset[i]
        if label[0][sel_class_idx]:
            if find_white_edge(img):
                num_edges_found = num_edges_found + 1
    return num_edges_found


def plot_histogram(a):
    """
    Plot histogram of RGB Pixel Intensities
    """
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(a)
    plt.axis('off')
    histo = plt.subplot(1, 2, 2)
    histo.set_ylabel('Count')
    histo.set_xlabel('Pixel Intensity')
    n_bins = 30
    plt.hist(a[:, :, 0].flatten(), bins=n_bins, lw=0, color='r', alpha=0.5)
    plt.hist(a[:, :, 1].flatten(), bins=n_bins, lw=0, color='g', alpha=0.5)
    plt.hist(a[:, :, 2].flatten(), bins=n_bins, lw=0, color='b', alpha=0.5)
    plt.show()


def plot_three_images(images):
    r = random.sample(images, 3)
    plt.figure(figsize=(16, 16))
    plt.subplot(131)
    plt.imshow(cv2.imread(r[0]))
    plt.subplot(132)
    plt.imshow(cv2.imread(r[1]))
    plt.subplot(133)
    plt.imshow(cv2.imread(r[2]))


def plot_keras_learning_curve():
    plt.figure(figsize=(10, 5))
    metrics = np.load('logs.npy', allow_pickle=True)[()]
    filt = ['acc']  # try to add 'loss' to see the loss learning curve
    for k in filter(lambda x: np.any([kk in x for kk in filt]), metrics.keys()):
        l = np.array(metrics[k])
        plt.plot(l, c='r' if 'val' not in k else 'b', label='val' if 'val' in k else 'train')
        x = np.argmin(l) if 'loss' in k else np.argmax(l)
        y = l[x]
        plt.scatter(x, y, lw=0, alpha=0.25, s=100, c='r' if 'val' not in k else 'b')
        plt.text(x, y, '{} = {:.4f}'.format(x, y), size='15', color='r' if 'val' not in k else 'b')
    plt.legend(loc=4)
    plt.axis([0, None, None, None])
    plt.grid()
    plt.xlabel('Number of epochs')
    plt.ylabel('Accuracy')
    plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          exp_name='',
                          model_path=''):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('Real class')
    plt.xlabel('Predicted class')
    plt.tight_layout()
    plt.savefig(f'./{model_path}/conf_mat_' + exp_name + '.png', dpi=300)
    # plt.show()
    return fig


def plot_learning_curve(history, exp_name, mdir):
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='lower right')
    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.savefig('./' + mdir + '/training_'+exp_name+'.png')
    # plt.show()


def plot_learning_info(data, dataH, dataC, name):
    plt.figure(figsize=(8, 5))
    plt.plot(data[name])
    plt.plot(dataH[name])
    plt.plot(dataC[name])
    plt.ylabel(name)
    plt.xlabel('epoch')
    plt.legend(['VGG', 'VGG16 + STD. HISTEQ', 'VGG16 + CLAHE'], loc='lower right')
    plt.tight_layout()
    plt.savefig('./' + model_dir + '/training_' + name + '_vgg16RMS_x_8000-968-im150_e50_batch64.png')
    plt.show()


def save_learning_info(history, exp_name, mdir):
    hist_df = pd.DataFrame(history.history)
    hist_csv_file = './' + mdir + '/training_' + exp_name + '_history.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)


def load_learning_info(mdir, exp_name):
    hist_csv_file = './' + mdir + '/training_' + exp_name + '_history.csv'
    df = pd.read_csv(hist_csv_file)
    return df


class MetricsCheckpoint(keras.callbacks.Callback):
    """Callback that saves metrics after each epoch"""

    def __init__(self, savepath):
        super(MetricsCheckpoint, self).__init__()
        self.savepath = savepath
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        np.save(self.savepath, self.history)


# def plot_class_distribution(map_characters, y_train):
#     dict_characters = map_characters
#     df = pd.DataFrame()
#     df["labels"] = y_train
#     lab = df['labels']
#     dist = lab.value_counts()
#     sns.countplot(lab)
#     print(dict_characters)


def plot_multiple_samples(subfolder):
    print(subfolder)
    # multipleImages = glob('%s\\Baza danych\\OCT2017\\train\\NORMAL\\**' % (getcwd()))
    multipleImages = glob(f'%s\\..\\DATA\\OCT2017\\train\\{subfolder}\\**' % (os.getcwd()))
    i_ = 0
    plt.rcParams['figure.figsize'] = (10.0, 10.0)
    plt.subplots_adjust(wspace=0, hspace=0)
    for im_list in multipleImages[:25]:
        im = cv2.imread(im_list)
        im = cv2.resize(im, (128, 128))
        plt.subplot(5, 5, i_ + 1)  # .set_title(l)
        plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        i_ += 1


if __name__ == '__main__':
    model_dir = 'models_dataset_8000-x'
    history_vgg = load_learning_info(model_dir, 'vgg16RMS_8000-968-im150_e50_batch64')
    history_vggH = load_learning_info(model_dir, 'vgg16RMS_HISTEQ_8000-968-im150_e50_batch64')
    history_vggC = load_learning_info(model_dir, 'vgg16RMS_CLAHE_8000-968-im150_e50_batch64')
    plot_learning_info(history_vgg, history_vggH, history_vggC, 'accuracy')
    plot_learning_info(history_vgg, history_vggH, history_vggC, 'loss')
    plot_learning_info(history_vgg, history_vggH, history_vggC, 'val_accuracy')
    plot_learning_info(history_vgg, history_vggH, history_vggC, 'val_loss')
