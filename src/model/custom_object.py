from tensorflow.keras.losses import Huber
from tensorflow.keras import backend as K

def apply_softmax_across_features(tensor):
    return K.softmax(tensor, axis=1)

def sum_features_per_instance(tensor):
    return K.sum(tensor, axis=1)

def get_last_timestep(tensor):
    return tensor[:, -1, :]

def get_sequence_output_and_cell_state(rnn_outputs):
    sequence_output = rnn_outputs[0]
    last_cell_state = rnn_outputs[2]
    return (sequence_output, last_cell_state)

def identity_operation(tensor):
    return tensor

custom_objects = {
    'Huber': Huber,
    'apply_softmax_across_features': apply_softmax_across_features,
    'sum_features_per_instance': sum_features_per_instance,
    'get_last_timestep': get_last_timestep,
    'get_sequence_output_and_cell_state': get_sequence_output_and_cell_state,
    'identity_operation': identity_operation,
}