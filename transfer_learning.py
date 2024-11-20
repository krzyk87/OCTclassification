# ----------- Step 1: Import Modules ---------------------------------
import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight
from os import getcwd
from keras.callbacks import EarlyStopping, ModelCheckpoint
from helper_functions import plot_histogram, plot_keras_learning_curve, plot_learning_curve, \
    plot_multiple_samples, save_learning_info
from my_network import pretrained_network
from data_loader import CustomGenerator
import wandb
import warnings
from my_wandb import MyWandbMetricsLogger
from model_param_configs import models_configs_LOCT, models_configs_CAVRI

warnings.filterwarnings("ignore")
print(tf.config.list_physical_devices('GPU'))
dataset_main = "OCT2017"

def train(config, dataset, model_type, num_epochs=50):    # dataset, model_type, optim, lr
    # Start a run, tracking hyperparameters

    run = wandb.init(config=config, project=f"{dataset_main}-project")
    #     # set the wandb project where this run will be logged
    #     project="oct5class-project",
    #
    #     # track hyperparameters and run metadata with wandb.config
    #     config={
    #         "optimizer": config["optimizer"],
    #         # "loss": "categorical_crossentropy",
    #         "dataset": dataset,
    #         "metric": "accuracy",
    #         "epochs": num_epochs,
    #         "batch_size": config["batch_size"],
    #         "learning_rate": config["learning_rate"],
    #         "model": model_type
    #     }
    # )
    # [optional] use wandb.config as your config
    wandb.config.update({"epochs": num_epochs, "model": model_type, "dataset": dataset})
    config = wandb.config
    new_run_name = (f'{model_type}-{config.optimizer}{config.learning_rate}-b{config.batch_size}-{dataset}-s'
                    + run.name.split('-')[-1])
    run.name = new_run_name.replace('.', '_')

    val_dir = f'../../DATA/{dataset}/val/'

    if 'OCT2017' in dataset:
        map_characters = {0: 'NORMAL', 1: 'CNV', 2: 'DME', 3: 'DRUSEN'}
        map_characters_inv = {'NORMAL': 0, 'CNV': 1, 'DME': 2, 'DRUSEN': 3}
        model_dir = 'models_dataset_8000-x'
        train_dir = f'../../DATA/{dataset}/train8000/'
    else:
        map_characters = {0: 'CNV', 1: 'DME', 2: 'DRUSEN', 3: 'NORMAL', 4: 'VMT'}
        map_characters_inv = {'CNV': 0, 'DME': 1, 'DRUSEN': 2, 'NORMAL': 3, 'VMT': 4}
        model_dir = 'models_dataset_CavriLOCT'
        train_dir = f'../../DATA/{dataset}/train/'
    batch_size = config.batch_size
    num_classes = len(map_characters)
    experiment_name = '_'.join(
        [dataset, model_type + config.optimizer, 'lr'+str(config.learning_rate), 'im'+str(imageSize),
         'e' + str(config.epochs), 'batch' + str(batch_size)])

    train_gen = CustomGenerator(train_dir, map_characters_inv, imageSize, batch_size=batch_size, shuffle=True, class_num=num_classes)
    val_gen = CustomGenerator(val_dir, map_characters_inv, imageSize, batch_size=8, class_num=num_classes)

    y_train = train_gen.get_labels()
    cls_w = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    cls_w = {i: cls_w[i] for i in range(num_classes)}

    # Define model
    new_model = pretrained_network(model_type, imageSize, num_classes, config.optimizer, config.learning_rate)
    # early_stop = EarlyStopping(monitor='val_accuracy', mode='max', patience=5, verbose=1)
    best_checkpoint = ModelCheckpoint(model_dir + '/' + experiment_name + '_best.hdf5', monitor="val_loss", mode="min",
                                      save_best_only=True, verbose=1)

    # Fit model
    history = new_model.fit(x=train_gen, epochs=config.epochs, validation_data=val_gen, class_weight=cls_w,
                            verbose=1, callbacks=[best_checkpoint, MyWandbMetricsLogger(log_freq="epoch")])
    new_model.save(model_dir + '/' + experiment_name + '.hdf5')
    plot_learning_curve(history, experiment_name, model_dir)
    save_learning_info(history, experiment_name, model_dir)

    wandb.alert(title='Run finished', text=f'Run {run.name} has finished!')
    wandb.finish()


# run individual trainings -----------------------------
if __name__ == '__main__':
    imageSize = 224
    model_name = 'vgg16'

    # config=None, dataset=dataset, model_type=model_name
    exp_config = models_configs_LOCT[model_name]
    for conf in exp_config:
        train(config=exp_config[conf], dataset=dataset_main, model_type=model_name)
        train(config=exp_config[conf], dataset=f'{dataset_main}_nowhite', model_type=model_name)
        train(config=exp_config[conf], dataset=f'{dataset_main}_HISTEQ', model_type=model_name)
        train(config=exp_config[conf], dataset=f'{dataset_main}_nowhite_HISTEQ', model_type=model_name)
        train(config=exp_config[conf], dataset=f'{dataset_main}_CLAHE', model_type=model_name)
        train(config=exp_config[conf], dataset=f'{dataset_main}_nowhite_CLAHE', model_type=model_name)

