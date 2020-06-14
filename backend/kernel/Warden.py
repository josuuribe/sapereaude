from kernel.Interfaces.IConsumer import IConsumer
from kernel.Interfaces.IProducer import IProducer
from kernel.Interfaces.IStoreManager import IStoreManager
from kernel.Neural import Neural
import kernel
from ctypes import c_int32
import multiprocessing
import numpy as np
import subprocess


class Warden:
    """
    This component is the central component that manages all dependencies among others.
    """

    def __init__(self, producer: IProducer, consumer: IConsumer, store_manager: IStoreManager):
        """
        Constructor that initializes entity
        :param producer: Producer that creates data.
        :param consumer: Consumer that consumes data.
        :param store_manager: Store manager to get and set data.
        """
        self.__producer__ = producer(self)
        self.__consumer__ = consumer(self)
        self.__store_manager__ = store_manager()
        self.__queue__ = multiprocessing.Queue()
        self.__event__ = multiprocessing.Event()
        self.__external_event__ = multiprocessing.Event()
        self.__internal_event__ = multiprocessing.Event()
        self.__command_id__: multiprocessing.Value = multiprocessing.Value(c_int32, 0)
        self.__user_id__: multiprocessing.Value = multiprocessing.Value(c_int32, 0)
        self.__threshold__ = kernel.get_threshold()
        self.__mode__ = 0
        self.__neural__ = None
        self.__commands__ = None

    def set_train_mode(self):
        """
        Set mode as train.
        :return:
        """
        self.__mode__ = 0

    def set_inference_mode(self):
        """
        Sets mode as inference.
        :return:
        """
        self.__mode__ = 1

    def is_train_mode(self):
        """
        Tells if actual mode is train.
        :return:
        """
        return self.__mode__ == 0

    def is_inference_mode(self):
        """
        Tells if actual mode is inference.
        :return:
        """
        return self.__mode__ == 1

    def get_active_command(self):
        """
        Get the active command, the command that is being trained.
        :return: The command in training mode.
        """
        return self.__command_id__

    def set_active_command(self, command_id):
        """
        Set the active command, the command that is going to be trained.
        :param command_id: Command id to be trained.
        """
        self.__command_id__ = command_id

    def get_active_user(self):
        """
        Get the user that is being trained.
        :return: The user in training mode.
        """
        return self.__user_id__

    def set_active_user(self, user_id):
        """
        Set the user that is in active mode.
        :param user_id: The user in training mode.
        :return: The active user.
        """
        self.__user_id__ = user_id

    def lock(self):
        """
        Locks the producer-consumer relation.
        """
        self.__consumer__.lock()

    def wait_internal(self):
        """
        Waits until a process be signaled.
        """
        self.__internal_event__.wait()

    def lock_internal_process(self):
        """
        Locks a process waiting to continue.
        :return:
        """
        self.__internal_event__.clear()

    def unlock_internal_process(self):
        """
        Unlocks a signaled process.
        """
        self.__internal_event__.set()

    def wait_process(self):
        """
        Waits until a process be signaled.
        """
        self.__external_event__.wait()

    def lock_external_process(self):
        """
        Locks a process waiting to continue.
        :return:
        """
        self.__external_event__.clear()

    def unlock_external_process(self):
        """
        Unlocks a signaled process.
        """
        self.__external_event__.set()

    def start(self):
        """
        Starts Warden and process associated depending on it is in train or inference the execution will be different.
        """
        print("Arrancando componente productor")
        producer_process = multiprocessing.Process(target=self.__producer__.start)
        producer_process.daemon = True
        producer_process.start()

        print("Arrancando componente consumidor")
        if self.is_inference_mode():
            print("Cargando datos del modelo")
            self.__store_manager__.refresh()
            self.__neural__ = Neural(self.__store_manager__)
            self.__neural__.load_model(self.__user_id__)
            print("Cargando datos del usuario")
            user = self.__store_manager__.get_user(self.__user_id__)
            commands = user.get_commands()
            self.__commands__ = {}
            for command in commands:
                parameters = command.get_parameters()
                if parameters is None:
                    parameters = [command.get_action()]
                else:
                    parameters.insert(0, command.get_action())
                self.__commands__[command.get_id()] = parameters
            self.__consumer__.start()
        else:
            self.__event__.clear()
            consumer_process = multiprocessing.Process(target=self.__consumer__.start)
            consumer_process.daemon = True
            consumer_process.start()

    def execute(self, data):
        """
        Executes the command infered.
        :param data: Data used for neural network to infer command.
        """
        try:
            command_id = self.__neural__.predict(data) + 1
            if command_id < len(self.__commands__):
                subprocess.run(self.__commands__[(command_id)[0]])
        except Exception as e:
            print(e)

    def train(self):
        """
        Executes the train mode
        """
        print("Preparando ML")
        self.__store_manager__.refresh()
        self.__neural__ = Neural(self.__store_manager__)
        user = self.__store_manager__.get_user(self.__user_id__)
        data = None
        target = np.array([])
        cmds = user.get_commands()
        for cmd in cmds:
            cmd = self.__store_manager__.load_command(self.__user_id__, cmd)
            if cmd is not None:
                d_res = cmd.get_eeg()
                if d_res is not None:
                    t_res = np.repeat(cmd.get_id(), len(d_res)).astype(np.int)
                    if data is None:
                        data = d_res
                    else:
                        if len(data.shape) == 1:
                            data = data[0].reshape(1, -1)
                        try:
                            data = np.append(data, d_res, axis=0)
                        except Exception:
                            print("exp")
                    target = np.concatenate((target, t_res), axis=0).astype(np.int)
        print("Empezando ML")
        self.__neural__.process(data, target)
        self.__neural__.create_softmax()
        train_loss = self.__neural__.reset_and_train_network(False)
        predicted_values, test_loss = self.__neural__.evaluate_network(self.__neural__.__data_train__,
                                                                       self.__neural__.__target_train__)
        percentage = self.__neural__.compute_success(self.__neural__.__target_train__, predicted_values) * 100
        print("Se ha conseguido un {0:.2f}% de acierto con un error de {1:.2f}% y una perdida de {2:.2f}.".format(
            percentage, test_loss, min(train_loss)))
        if percentage > self.__threshold__:
            print("Se va a guardar este modelo.")
            self.__neural__.save_model(self.__user_id__)
        else:
            print("La calidad es demasiado baja y se va a descartar este modelo.")
        self.unlock_external_process()

    def stop(self):
        """
        Stops Warden and all processes.
        """
        self.__producer__.stop()
        self.__consumer__.stop()
