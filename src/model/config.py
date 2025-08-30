# Loss function
DELTA = 0.0001

# BATCH_SIZE and NUM_EPOCHS
BATCH_SIZE = 30
EPOCHS = 50

regressor_config = {
    "ANNR": {
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS": EPOCHS,
        "NUM_NODES": 100
    },
    "LSTM": {
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS": EPOCHS,
        "NUM_UNITS": 100
    },
    "GRU": {
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS": EPOCHS,
        "NUM_UNITS": 100
    },
    "LSTMA": {
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS": EPOCHS,
        "NUM_UNITS": 100,
        "ATTENTION_UNITS": 32
    },
    "TCN": {
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS": EPOCHS,
        "NUM_FILTERS": 64,
        "KERNEL_SIZE": 3,
        "DILATIONS": (1, 2, 4, 8),
        "DROPOUT_RATE": 0.1,
        "TCN_ACTIVATION": "relu",
        "USE_SPATIAL_DROPOUT": True
    },
    "TCNA": {
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS": EPOCHS,
        "NUM_FILTERS": 64,
        "KERNEL_SIZE": 3,
        "DILATIONS": (1, 2, 4, 8),
        "DROPOUT_RATE": 0.1,
        "TCN_ACTIVATION": "relu",
        "USE_SPATIAL_DROPOUT": True,
        "ATTENTION_UNITS": 32
    }
}

classifier_config = {
    "ANNC": {
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS": EPOCHS,
        "NUM_NODES": 100
    },
    "CNNA": {
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS": EPOCHS,
        "NUM_FILTERS": 64,
        "KERNEL_SIZE": 3,
        "ATTENTION_UNITS": 32,
        "DROPOUT_RATE": 0.3
    },
    "ResNet1D": {
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS": EPOCHS,
        "FILTERS": 64,
        "BLOCKS": 3,
        "KERNEL_SIZE": 3,
        "DROPOUT_RATE": 0.3
    },
    "MLP": {
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS": EPOCHS,
        "NUM_LAYERS": 4,
        "NUM_UNITS": 128,
        "DROPOUT_RATE": 0.4
    }
}

ensembler_config = {
    "RF": {
        "NUM_ESTIMATORS": 250,
        "RANDOM_STATE": 190,
        "MAX_DEPTH": 37
    },
    "XGB": {
        "NUM_ESTIMATORS": 250,
        "MAX_DEPTH": 37,
        "LEARNING_RATE": 0.05,
        "USE_LABEL_ENCODER": False,
        "EVAL_METRIC": "logloss"
    },
    "LGBM": {
        "NUM_ESTIMATORS": 250,
        "MAX_DEPTH": 37,
        "LEARNING_RATE": 0.05
    },
    "CB": {
        "ITERATIONS": 250,
        "DEPTH": 15,
        "LEARNING_RATE": 0.05
    }
}