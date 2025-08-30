from keras import backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Input, Lambda, Multiply, Activation, Conv1D, Add, SpatialDropout1D
from tensorflow.keras.losses import Huber
import os

from model.custom_object import *
from model.config import DELTA, regressor_config as config

def lstm_driver(x_train, y_train, dump_file, batch_size=config["LSTM"]["BATCH_SIZE"], epochs=config["LSTM"]["EPOCHS"], num_units=config["LSTM"]["NUM_UNITS"]):
    input_shape = (x_train.shape[1], x_train.shape[2])

    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(units=num_units, return_sequences=True, activation='relu'))
    model.add(LSTM(units=num_units, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(num_units, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='tanh'))

    loss_fn = Huber(delta=DELTA)
    model.compile(optimizer='adam', loss = loss_fn)

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)

    path = os.path.join(dump_file, f"model.keras")
    model.save(path)
    return model, path

def gru_driver(x_train, y_train, dump_file, batch_size=config["GRU"]["BATCH_SIZE"], epochs=config["GRU"]["EPOCHS"], num_units=config["GRU"]["NUM_UNITS"]):
    input_shape = (x_train.shape[1], x_train.shape[2])

    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(GRU(units=num_units, return_sequences=True, activation='relu'))
    model.add(GRU(units=num_units, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(units=num_units, activation='tanh'))
    model.add(Dense(units=1, activation='tanh'))
    loss_fn = Huber(delta=DELTA)
    model.compile(optimizer='adam', loss=loss_fn)

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)

    path = os.path.join(dump_file, f"model.keras")
    model.save(path)
    return model, path

def lstm_attention_driver(x_train, y_train, dump_file, batch_size=config["LSTMA"]["BATCH_SIZE"], epochs=config["LSTMA"]["EPOCHS"], num_units=config["LSTMA"]["NUM_UNITS"]):
    input_shape = (x_train.shape[1], x_train.shape[2])
    inputs = Input(shape=input_shape)
    lstm_out = LSTM(num_units, return_sequences=True, name='lstm_layer')(inputs)
    attention_scores = Dense(1, activation='tanh', name='attention_score_vec')(lstm_out)
    attention_weights = Lambda(
        apply_softmax_across_features,
        output_shape=identity_operation,
        name='attention_weights'
    )(attention_scores)
    weighted_lstm_out = Multiply(name='weighted_lstm_outputs')([lstm_out, attention_weights])
    context_vector = Lambda(
        sum_features_per_instance,
        output_shape=get_sequence_output_and_cell_state,
        name='context_vector'
    )(weighted_lstm_out)
    outputs = Dense(1, activation='linear', name='output_layer')(context_vector)

    loss_fn = Huber(delta=DELTA)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss=loss_fn)

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
    path = os.path.join(dump_file, f"model.keras")
    model.save(path)
    return model, path

def residual_block(x, dilation_rate, nb_filters, kernel_size, dropout_rate, activation='relu', use_spatial_dropout=True):
    prev_x = x

    conv1 = Conv1D(filters=nb_filters,
                   kernel_size=kernel_size,
                   dilation_rate=dilation_rate,
                   padding='causal', 
                   name=f'res_block_d{dilation_rate}_conv1')(x)
    act1 = Activation(activation, name=f'res_block_d{dilation_rate}_act1')(conv1)
    if use_spatial_dropout:
        drop1 = SpatialDropout1D(dropout_rate, name=f'res_block_d{dilation_rate}_sdrop1')(act1)
    else:
        drop1 = Dropout(dropout_rate, name=f'res_block_d{dilation_rate}_drop1')(act1)

    conv2 = Conv1D(filters=nb_filters,
                   kernel_size=kernel_size,
                   dilation_rate=dilation_rate,
                   padding='causal', 
                   name=f'res_block_d{dilation_rate}_conv2')(drop1)
    act2 = Activation(activation, name=f'res_block_d{dilation_rate}_act2')(conv2)
    if use_spatial_dropout:
        drop2 = SpatialDropout1D(dropout_rate, name=f'res_block_d{dilation_rate}_sdrop2')(act2)
    else:
        drop2 = Dropout(dropout_rate, name=f'res_block_d{dilation_rate}_drop2')(act2)

    
    input_channels = K.int_shape(prev_x)[-1]
    if input_channels != nb_filters:
        shortcut = Conv1D(filters=nb_filters,
                          kernel_size=1,
                          padding='same',
                          name=f'res_block_d{dilation_rate}_shortcut')(prev_x)
    else:
        shortcut = prev_x

    res_x = Add(name=f'res_block_d{dilation_rate}_add')([shortcut, drop2])

    return res_x


def tcn_driver(x_train, y_train, dump_file,
               batch_size=config["TCN"]["BATCH_SIZE"], 
               epochs=config["TCN"]["EPOCHS"], 
               num_filters=config["TCN"]["NUM_FILTERS"],
               kernel_size=config["TCN"]["KERNEL_SIZE"], 
               dilations=config["TCN"]["DILATIONS"], 
               dropout_rate=config["TCN"]["DROPOUT_RATE"], 
               tcn_activation=config["TCN"]["TCN_ACTIVATION"],
               use_spatial_dropout=config["TCN"]["USE_SPATIAL_DROPOUT"]):

    input_shape = (x_train.shape[1], x_train.shape[2]) # (time_steps, features)

    inputs = Input(shape=input_shape)
    x = inputs
    for i, d in enumerate(dilations):
        x = residual_block(x,
                           dilation_rate=d,
                           nb_filters=num_filters,
                           kernel_size=kernel_size,
                           dropout_rate=dropout_rate,
                           activation=tcn_activation,
                           use_spatial_dropout=use_spatial_dropout)
    output_slice = Lambda(
        get_last_timestep,
        output_shape=get_sequence_output_and_cell_state,
        name='slice_last_timestep'
    )(x)
    dense1 = Dense(num_filters // 2, activation='tanh', name='dense_1')(output_slice)
    drop_dense1 = Dropout(dropout_rate, name='dropout_dense_1')(dense1)
    outputs = Dense(1, activation='linear', name='output_dense')(drop_dense1)

    model = Model(inputs=inputs, outputs=outputs)

    loss_fn = Huber(delta=DELTA)
    model.compile(optimizer='adam', loss=loss_fn)

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
    path = os.path.join(dump_file, f"model.keras")
    model.save(path)
    return model, path

def tcn_attention_driver(x_train, y_train, dump_file,
                         batch_size=config["TCNA"]["BATCH_SIZE"],
                         epochs=config["TCNA"]["EPOCHS"],
                         num_filters=config["TCNA"]["NUM_FILTERS"],
                         kernel_size=config["TCNA"]["KERNEL_SIZE"],
                         dilations=config["TCNA"]["DILATIONS"],
                         dropout_rate=config["TCNA"]["DROPOUT_RATE"],
                         tcn_activation=config["TCNA"]["TCN_ACTIVATION"],
                         use_spatial_dropout=config["TCNA"]["USE_SPATIAL_DROPOUT"],
                         attention_units=config["TCNA"]["ATTENTION_UNITS"]):

    input_shape = (x_train.shape[1], x_train.shape[2])
    inputs = Input(shape=input_shape)
    x = inputs

    for i, d in enumerate(dilations):
        x = residual_block(x,
                           dilation_rate=d,
                           nb_filters=num_filters,
                           kernel_size=kernel_size,
                           dropout_rate=dropout_rate,
                           activation=tcn_activation,
                           use_spatial_dropout=use_spatial_dropout)
    tcn_output = x

    attention_energy = Dense(attention_units, activation='tanh', name='attention_energy')(tcn_output)
    attention_scores = Dense(1, activation='linear', name='attention_score_vec')(attention_energy)
    attention_weights = Activation(apply_softmax_across_features, name='attention_weights')(attention_scores)

    weighted_sequence = Multiply(name='attention_weighted_sequence')([tcn_output, attention_weights])

    context_vector = Lambda(
        sum_features_per_instance,
        output_shape=get_sequence_output_and_cell_state,
        name='attention_context_vector'
    )(weighted_sequence)

    dense1 = Dense(num_filters // 2, activation='tanh', name='dense_1')(context_vector)
    drop_dense1 = Dropout(dropout_rate, name='dropout_dense_1')(dense1)
    outputs = Dense(1, activation='linear', name='output_dense')(drop_dense1)

    model = Model(inputs=inputs, outputs=outputs)

    loss_fn = Huber(delta=DELTA)
    model.compile(optimizer='adam', loss=loss_fn)

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
    path = os.path.join(dump_file, f"model.keras")
    model.save(path)
    return model, path

def ann_regression_driver(x_train, y_train, dump_file, 
                          batch_size=config["ANNR"]["BATCH_SIZE"], 
                          epochs=config["ANNR"]["EPOCHS"], 
                          num_nodes=config["ANNR"]["NUM_NODES"]):
    model = Sequential()
    model.add(Dense(num_nodes, input_dim = (x_train.shape[1]), activation='relu'))
    model.add(Dense(num_nodes, activation='relu'))
    model.add(Dense(units=1, activation='tanh'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size, epochs, verbose=1)
    path = os.path.join(dump_file, f"model.keras")
    model.save(path)
    return model, path

regressor_drivers = {
    "ANNR": (ann_regression_driver, False),
    "LSTM": (lstm_driver, True),
    "TCN": (tcn_driver, True),
    "GRU": (gru_driver, True),
    "LSTMA": (lstm_attention_driver, True),
    "TCNA": (tcn_attention_driver, True),
}