models_configs_CAVRI = {
    "vgg16":{
        "Adam": {"optimizer": "Adam", "batch_size": 16, "learning_rate": 0.0002,},
        "RMS": {"optimizer": "RMS", "batch_size": 16, "learning_rate": 0.0005,},
        "SGD": {"optimizer": "SGD", "batch_size": 16, "learning_rate": 0.001,}
    },
    "inception": {
        "Adam": {"optimizer": "Adam", "batch_size": 16, "learning_rate": 0.0001,},
        "RMS": {"optimizer": "RMS", "batch_size": 16, "learning_rate": 0.0002,},
        "SGD": {"optimizer": "SGD", "batch_size": 16, "learning_rate": 0.0005,}
    },
    "xception": {
        "Adam": {"optimizer": "Adam", "batch_size": 16, "learning_rate": 0.00005,},
        "RMS": {"optimizer": "RMS", "batch_size": 16, "learning_rate": 0.00005,},
        "SGD": {"optimizer": "SGD", "batch_size": 16, "learning_rate": 0.001,}
    },
    "resnet50": {
        "Adam": {"optimizer": "Adam", "batch_size": 16, "learning_rate": 0.0005,},
        "RMS": {"optimizer": "RMS", "batch_size": 16, "learning_rate": 0.00001,},
        "SGD": {"optimizer": "SGD", "batch_size": 16, "learning_rate": 0.0001,}
    },
    "densenet121": {
        "Adam": {"optimizer": "Adam", "batch_size": 16, "learning_rate": 0.0001,},
        "RMS": {"optimizer": "RMS", "batch_size": 16, "learning_rate": 0.00001,},
        "SGD": {"optimizer": "SGD", "batch_size": 16, "learning_rate": 0.0001,}
    },
}

models_configs_LOCT = {
    "vgg16":{
        "Adam": {"optimizer": "Adam", "batch_size": 16, "learning_rate": 0.00001,},
        "RMS": {"optimizer": "RMS", "batch_size": 16, "learning_rate": 0.00001,},
        "SGD": {"optimizer": "SGD", "batch_size": 16, "learning_rate": 0.001,}
    },
    "inception": {
        "Adam": {"optimizer": "Adam", "batch_size": 16, "learning_rate": 0.00001,},
        "RMS": {"optimizer": "RMS", "batch_size": 16, "learning_rate": 0.00001,},
        "SGD": {"optimizer": "SGD", "batch_size": 16, "learning_rate": 0.0001,}
    },
    "xception": {
        "Adam": {"optimizer": "Adam", "batch_size": 16, "learning_rate": 0.00005,},
        "RMS": {"optimizer": "RMS", "batch_size": 16, "learning_rate": 0.0005,},
        "SGD": {"optimizer": "SGD", "batch_size": 16, "learning_rate": 0.001,}
    },
    "resnet50": {
        "Adam": {"optimizer": "Adam", "batch_size": 32, "learning_rate": 0.0001,},
        "RMS": {"optimizer": "RMS", "batch_size": 8, "learning_rate": 0.00005,},
        "SGD": {"optimizer": "SGD", "batch_size": 8, "learning_rate": 0.00005,}
    },
    "densenet121": {
        # "Adam": {"optimizer": "Adam", "batch_size": 16, "learning_rate": 0.00005,},
        # "RMS": {"optimizer": "RMS", "batch_size": 16, "learning_rate": 0.00001,},
        "SGD": {"optimizer": "SGD", "batch_size": 16, "learning_rate": 0.0002,}
    },
}
