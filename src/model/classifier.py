from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Lambda, Multiply, Activation, BatchNormalization, Conv1D, Add, GlobalAveragePooling1D
import os
from model.custom_object import *
from model.config import classifier_config as config

def ann_classifier_driver(x_train, y_train, dump_file, 
                          batch_size=config["ANNC"]["BATCH_SIZE"], 
                          epochs=config["ANNC"]["EPOCHS"], 
                          num_nodes=config["ANNC"]["NUM_NODES"]):
    model = Sequential()
    model.add(Dense(num_nodes, input_dim = (x_train.shape[1]), activation='relu'))
    model.add(Dense(num_nodes, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.fit(x_train, y_train, batch_size, epochs, verbose=1)

    path = os.path.join(dump_file, f"model.keras")
    model.save(path)
    return model, path

def cnn_attention_classifier_driver(x_train, y_train, dump_file,
                                    batch_size=config["CNNA"]["BATCH_SIZE"],
                                    epochs=config["CNNA"]["EPOCHS"],
                                    num_filters=config["CNNA"]["NUM_FILTERS"],
                                    kernel_size=config["CNNA"]["KERNEL_SIZE"],
                                    attention_units=config["CNNA"]["ATTENTION_UNITS"],
                                    dropout_rate=config["CNNA"]["DROPOUT_RATE"]):
    input_shape = (x_train.shape[1], x_train.shape[2])
    inputs = Input(shape=input_shape)

    x = Conv1D(filters=num_filters, kernel_size=kernel_size, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    attention_scores = Dense(attention_units, activation='tanh')(x)
    attention_weights = Dense(1, activation='softmax')(attention_scores)
    weighted_output = Multiply()([x, attention_weights])
    context_vector = Lambda(sum_features_per_instance)(weighted_output)

    x = Dense(64, activation='relu')(context_vector)
    x = Dropout(dropout_rate)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, output)

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)

    path = os.path.join(dump_file, f"model.keras")
    model.save(path)
    return model, path

def resnet1d_classifier_driver(x_train, y_train, dump_file,
                               batch_size=config["ResNet1D"]["BATCH_SIZE"],
                               epochs=config["ResNet1D"]["EPOCHS"],
                               filters=config["ResNet1D"]["FILTERS"], 
                               kernel_size=config["ResNet1D"]["KERNEL_SIZE"], 
                               blocks=config["ResNet1D"]["BLOCKS"],
                               dropout_rate=config["ResNet1D"]["DROPOUT_RATE"]):
    inputs = Input(shape=(x_train.shape[1], x_train.shape[2]))
    x = inputs

    for i in range(blocks):
        shortcut = x
        x = Conv1D(filters, kernel_size, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv1D(filters, kernel_size, padding='same')(x)
        x = BatchNormalization()(x)
        x = Add()([shortcut, x])
        x = Activation('relu')(x)
        x = Dropout(dropout_rate)(x)

    x = GlobalAveragePooling1D()(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
    path = os.path.join(dump_file, f"model.keras")
    model.save(path)
    return model, path

def deep_mlp_classifier_driver(x_train, y_train, dump_file,
                               batch_size=config["MLP"]["BATCH_SIZE"],
                               epochs=config["MLP"]["EPOCHS"],
                               num_layers=config["MLP"]["NUM_LAYERS"], 
                               num_units=config["MLP"]["NUM_UNITS"],
                               dropout_rate=config["MLP"]["DROPOUT_RATE"]):
    inputs = Input(shape=(x_train.shape[1],))
    x = inputs
    for _ in range(num_layers):
        x = Dense(num_units, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
    path = os.path.join(dump_file, f"model.keras")
    model.save(path)
    return model, path

classifier_drivers = {
    "ANNC": (ann_classifier_driver, False),
    "MLP": (deep_mlp_classifier_driver, False),
    "CNNA": (cnn_attention_classifier_driver, True),
    "ResNet1D": (resnet1d_classifier_driver, True),
}