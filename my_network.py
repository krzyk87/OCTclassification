import os.path
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.resnet import ResNet50
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, Flatten
from keras.models import Model
import keras.optimizers


def pretrained_network(model_type, image_size, numclasses, optimizer_type, lr):   # train_gen
    if model_type == 'vgg16':
        base_model = VGG16(weights="imagenet", include_top=False, input_shape=(image_size, image_size, 3))
    elif model_type == 'inception':
        base_model = InceptionV3(weights="imagenet", include_top=False, input_shape=(image_size, image_size, 3))
    elif model_type == 'xception':
        base_model = Xception(weights="imagenet", include_top=False, input_shape=(image_size, image_size, 3))
    elif model_type == 'resnet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
    elif model_type == 'densenet121':
        base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
    else:
        print('Unsupported model type!!!!')
        return None
    print(f'Running model: --------{model_type}--------')

    if 'Adam' in optimizer_type:
        optimizer = keras.optimizers.Adam(learning_rate=lr)     # 0.001
    elif 'RMS' in optimizer_type:
        optimizer = keras.optimizers.RMSprop(lr=lr)             # 0.0001
    elif 'SGD' in optimizer_type:
        optimizer = keras.optimizers.SGD(learning_rate=lr)      # 0.001
    else:
        # optimizer = keras.optimizers.Adadelta(learning_rate=lr) # 0.001
        print('Unsupported optimizer!!!!')
        return None

    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
    # Add top layer
    x = base_model.output
    x = Flatten()(x)
    predictions = Dense(numclasses, activation='softmax')(x)
    new_model = Model(inputs=base_model.input, outputs=predictions)

    new_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    # new_model.summary()

    return new_model
