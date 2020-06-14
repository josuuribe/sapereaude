import sys

sys.path.extend(['E:\\Proyectos\\sapereaude', 'E:\\Proyectos\\sapereaude\\backend', 'E:/Proyectos/sapereaude'])
import json
import os
import backend

path = os.path.join(backend.CONFIG_PATH)

with open(path) as json_file:
    try:
        data = json.load(json_file)
    except Exception:
        print("No se encuentra fichero de configuración.")


def get_connection_string():
    return data['DataSource']['path']


def get_sample_rate():
    return data['EEG']['sample_rate']


def get_channels():
    return data['EEG']['channels']


def get_window_size():
    return data['EEG']['window_size']


def get_number_blocks():
    return data['EEG']['number_blocks']


def get_bands():
    bands = data['EEG']['bands']
    default_bands = {}
    for band in bands:
        freqs = []
        for values in bands[band]:
            freqs.append(bands[band][values])
        default_bands[band] = freqs


def get_learning_rate():
    return data['ML']['learning_rate']


def get_epochs():
    return data['ML']['epochs']


def get_batch_size():
    return data['ML']['batch_size']


def get_threshold():
    return data['ML']['threshold']
