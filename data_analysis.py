import os.path

import matplotlib.pyplot as plt
import numpy as np
from data_loader import CustomGenerator
from helper_functions import calc_hist_data, count_white_edge_in_subset, remove_white_edge

# setup data parameters
sel_dataset = 'LOCT'
# selected_class = 'NORMAL'
histogram_analysis = True
imageSize = 224     # 150 / 224 / None

data_dir = ''
map_characters_inv = {}
data_dir = '../../DATA/OCT2017_nowhite_CLAHE/'
map_characters_inv = {'NORMAL': 0, 'CNV': 1, 'DME': 2, 'DRUSEN': 3}
train_dir = data_dir + 'train8000/'

val_dir = data_dir + 'val/'
test_dir = data_dir + "test/"

disease_list = {"CNV", "DME", "DRUSEN", "NORMAL"}

for selected_class in disease_list:
    sel_class_idx = map_characters_inv[selected_class]
    # get data ----------------------------------
    train_gen = CustomGenerator(train_dir, map_characters_inv, imageSize, batch_size=1, class_num=len(map_characters_inv),
                                subset=selected_class)
    val_gen = CustomGenerator(val_dir, map_characters_inv, imageSize, batch_size=1, class_num=len(map_characters_inv),
                              subset=selected_class)
    test_gen = CustomGenerator(test_dir, map_characters_inv, imageSize, batch_size=1, class_num=len(map_characters_inv),
                               subset=selected_class)

    # find white edges --------------------------
    num_edges_train = count_white_edge_in_subset(train_gen, sel_class_idx)
    num_edges_val = count_white_edge_in_subset(val_gen, sel_class_idx)
    num_edges_test = count_white_edge_in_subset(test_gen, sel_class_idx)
    print(f'{num_edges_train}/{len(train_gen)} ({num_edges_train/len(train_gen):.2f}) white edges found in {selected_class} train set')
    print(f'{num_edges_val}/{len(val_gen)} ({num_edges_val/len(val_gen):.2f}) white edges found in {selected_class} val set')
    print(f'{num_edges_test}/{len(test_gen)} ({num_edges_test/len(test_gen):.2f}) white edges found in {selected_class} test set')
    sum_white = num_edges_train+num_edges_val+num_edges_test
    sum_img = len(train_gen)+len(val_gen)+len(test_gen)

    print(f'{sum_white}/{sum_img} ({sum_white/sum_img}) in total')

    # remove white edges ------------------------
    remove_white_edge(train_gen, sel_class_idx)
    remove_white_edge(val_gen, sel_class_idx)
    remove_white_edge(test_gen, sel_class_idx)

    # histogram analysis -------------------------
    if histogram_analysis:
        histgram_data = np.zeros(256, dtype=np.int64)
        histgram_data = calc_hist_data(train_gen, histgram_data, sel_class_idx)
        histgram_data = calc_hist_data(val_gen, histgram_data, sel_class_idx)
        histgram_data = calc_hist_data(test_gen, histgram_data, sel_class_idx)
        max_hist = np.max(histgram_data)
        histgram_data = np.divide(histgram_data, max_hist)

        # plot histogram -----------------------------
        plt.bar(np.arange(256), histgram_data)
        plt.title(f'{data_dir.split("/")[-2]} ({selected_class})')
        plt.xlabel('Pixel intensity')
        plt.savefig(os.path.join(data_dir, selected_class + f'_{imageSize}_hist.png'))
        plt.show()
