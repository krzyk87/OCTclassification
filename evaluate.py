# ----------- Step 1: Import Modules ---------------------------------
import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from keras.utils.np_utils import to_categorical
import tensorflow as tf
from tensorflow.python.eager.context import device

from helper_functions import plot_confusion_matrix
from data_loader import get_data, CustomGenerator
from model_param_configs import models_configs_LOCT
import wandb

print(tf.config.list_physical_devices('GPU'))

# ----------- Step 2: Load Data --------------------------------------

imageSize = 224     # 150 / 224
numepochs = 50
dataset_main = "OCT2017"


def test(dataset, model_type, config, is_best):
    optimizer = config['optimizer']
    batch_size = config['batch_size']
    lr = config['learning_rate']
    test_dir = f"../../DATA/{dataset}/test/"

    if 'OCT2017' in dataset:
        map_characters = {0: 'NORMAL', 1: 'CNV', 2: 'DME', 3: 'DRUSEN'}
        map_characters_inv = {'NORMAL': 0, 'CNV': 1, 'DME': 2, 'DRUSEN': 3}
        model_dir = f'models_dataset_8000-x/{model_type}'
    else:
        map_characters = {0: 'CNV', 1: 'DME', 2: 'DRUSEN', 3: 'NORMAL', 4: 'VMT'}
        map_characters_inv = {'CNV': 0, 'DME': 1, 'DRUSEN': 2, 'NORMAL': 3, 'VMT': 4}
        model_dir = f'models_dataset_CavriLOCT/{model_type}'
    params = [dataset, model_type + optimizer, 'lr'+str(lr), 'im'+str(imageSize), 'e' + str(numepochs), 'batch' + str(batch_size)]
    if is_best:
        params.append('best')
    experiment_name = '_'.join(params)
    print(f'Testing experiment: {experiment_name}')

    wandb.init(project=f"{dataset_main}-project", name=f"test_{experiment_name}")

    num_classes = len(map_characters)
    model_path = f'./{model_dir}/' + experiment_name + '.hdf5'

    x_test, y_test = get_data(test_dir, map_characters_inv, imageSize)
    y_test_hot = to_categorical(y_test, num_classes=num_classes)

    model2test = keras.models.load_model(model_path)
    optim = None
    if optimizer == 'Adam':
        optim = keras.optimizers.Adam(),
    elif optimizer == 'RMS':
        optim = keras.optimizers.RMSprop(),
    elif optimizer == 'SGD':
        optim = keras.optimizers.SGD()
    model2test.compile(loss='categorical_crossentropy',
                       optimizer=optim,  # Adam() / SGD() / RMSprop()
                       metrics=['accuracy'])

    y_pred = model2test.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test_hot, axis=1)

    # classification report --------------
    print('\n', sklearn.metrics.classification_report(y_true, y_pred_classes,
                                                      target_names=list(map_characters_inv.keys()), digits=4), sep='')
    with open(f'{model_dir}/test_{experiment_name}.txt', 'w') as f:
        f.write(sklearn.metrics.classification_report(y_true, y_pred_classes,
                                                      target_names=list(map_characters_inv.keys()), digits=4))
    class_report_dict = sklearn.metrics.classification_report(y_true, y_pred_classes,
                                                              target_names=list(map_characters_inv.keys()), digits=4,
                                                              output_dict=True)
    wandb.log({"test/accuracy": class_report_dict['accuracy'],
               "test/precision": class_report_dict['macro avg']['precision'],
               "test/recall": class_report_dict['macro avg']['recall'],
               "test/f1": class_report_dict['macro avg']['f1-score'],})

    # confusion matrix -------------------
    confusion_mtx = confusion_matrix(y_true, y_pred_classes)
    conf_mat_fig = plot_confusion_matrix(confusion_mtx, classes=list(map_characters_inv.keys()), cmap="terrain_r",
                          exp_name=experiment_name, model_path=model_dir)
    wandb.log({f"conf mat {model_type} {optimizer}": conf_mat_fig})
    # wandb.sklearn.plot_confusion_matrix(y_true, y_pred_classes, labels=list(map_characters_inv.keys()),)
    wandb.log({f"my_conf_mat_id_{model_type}_{optimizer}": wandb.plot.confusion_matrix(preds=y_pred_classes, y_true=y_true,
                                                             class_names=list(map_characters_inv.keys()))})

    wandb.finish()


if __name__ == '__main__':
    model_name = 'densenet121'
    best = False
    exp_config = models_configs_LOCT[model_name]
    for conf in exp_config:
        test(dataset_main, model_name, exp_config[conf], best)
        test(f'{dataset_main}_nowhite', model_name, exp_config[conf], best)
        test(f'{dataset_main}_HISTEQ', model_name, exp_config[conf], best)
        test(f'{dataset_main}_nowhite_HISTEQ', model_name, exp_config[conf], best)
        test(f'{dataset_main}_CLAHE', model_name, exp_config[conf], best)
        test(f'{dataset_main}_nowhite_CLAHE', model_name, exp_config[conf], best)
