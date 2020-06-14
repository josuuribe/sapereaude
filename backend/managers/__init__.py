from kernel.Interfaces import IStoreManager, IProducer, IConsumer
import pkgutil
import importlib
import inspect
import managers
import os
import json
import backend

global __processed__

__processed__ = False
__store_manager__ = None
__consumer_manager__ = None
__producer_manager__ = None

if not __processed__:
    imported_package = __import__(managers.__name__, fromlist=['blah'])

    ui_manager = None

    for _, pluginname, ispkg in pkgutil.iter_modules(imported_package.__path__):
        name = managers.__name__ + "." + pluginname
        plugin_module = importlib.import_module(name)
        for (key, value) in inspect.getmembers(plugin_module, inspect.isclass):
            if issubclass(value, IStoreManager.IStoreManager) & (value is not IStoreManager.IStoreManager):
                # print("Encontrado IUserManager: "+str(value))
                __store_manager__ = value
            if issubclass(value, IConsumer.IConsumer) & (value is not IConsumer.IConsumer):
                # print("Encontrado IConsumer: "+str(value))
                __consumer_manager__ = value
            if issubclass(value, IProducer.IProducer) & (value is not IProducer.IProducer):
                # print("Encontrado IProducer: "+str(value))
                __producer_manager__ = value

    data = None
    path = os.path.join(backend.CONFIG_PATH)
    with open(path) as json_file:
        data = json.load(json_file)

    __processed__ = True


def get_store_manager():
    return __store_manager__


def get_consumer_manager():
    return __consumer_manager__


def get_producer_manager():
    return __producer_manager__


def get_host():
    return data['CyKitServer']['host']


def get_port():
    return data['CyKitServer']['port']


def get_buffer_size():
    return data['CyKitServer']['buffer_size']


def get_delay():
    return data['Consumer']['delay']


def get_duration():
    return data['Consumer']['duration']
