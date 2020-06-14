from datetime import datetime, timedelta
import managers
from backend.kernel import EEG
from kernel.Interfaces.IConsumer import IConsumer
from kernel.Warden import Warden
import numpy as np
import traceback
import sys
from multiprocessing import Value
from ctypes import c_bool
import kernel


class TimeConsumer(IConsumer):

    def __init__(self, warden: Warden):
        """
        Constructor that initializes entity
        :param warden: Warden to be used.
        """
        self.eeg = EEG.EEG()
        self.data: np.array = None
        self.warden = warden
        self.current_time = datetime.now()
        self.lapse = timedelta(milliseconds=managers.get_delay())
        self.__stop__ = False
        self.__stop__ = Value(c_bool, self.__stop__)
        self.__size__ = kernel.get_window_size()

    def lock(self):
        """
        Manages if this consumer can start to consume or not.
        :return:
        """
        now = datetime.now()
        lapse_time = now - self.current_time
        if lapse_time > self.lapse and not self.__stop__.value:
            self.current_time = now
            self.warden.__event__.set()

    def start(self):
        """
        Starts consumer in training or inference mode.
        """
        self.__stop__.value = False
        if self.warden.is_train_mode():
            self.__start_train__()
        elif self.warden.is_inference_mode():
            try:
                self.__start_inference__()
            except KeyboardInterrupt:
                return
        else:
            raise Exception("Invalid mode")

    def __start_inference__(self):
        """
        Starts inference mode.
        """
        rows = 0
        first_zero = False
        self.data = np.array([])
        while not self.__stop__.value:
            try:
                self.warden.__event__.wait()
                while self.warden.__event__.is_set() and len(self.data) < self.__size__:
                    try:
                        message = self.warden.__queue__.get()
                        if first_zero:
                            if message[0] == 0:
                                rows = 0
                            else:
                                rows = rows + 1
                            self.data = np.vstack([self.data, message])
                        elif int(message[0]) == 0:
                            self.data = message
                            first_zero = True
                            rows = 0
                    except Exception as e:
                        if first_zero:
                            self.data = self.data[: -(rows + 1)]
                            rows = 0
                        print(e)
                        traceback.print_exc()

                exc = len(self.data) % self.__size__
                if exc > 0:
                    rest = self.data[-exc:]
                    self.data = self.data[:-exc]
                else:
                    rest = np.array([])
                    first_zero = False
                self.eeg.add(self.data)
                total = self.run_eeg()
                neural_data = list()
                neural_data.append(total)
                self.warden.execute(neural_data)
                self.data = rest
                rows = len(self.data) - 1
                self.warden.__event__.clear()
            except Exception as e:
                print(e)
                self.data = np.array([])
                first_zero = False
                rows = 0
                traceback.print_exc()
                self.eeg.reset()
        print("\r\nConsumidor termina")

    def __start_train__(self):
        """
        Starts train mode.
        """
        rows = 0
        first_zero = False
        self.data = np.array([])
        self.warden.lock_internal_process()
        print("¡Vamos!")
        while not self.__stop__.value or self.warden.__queue__.qsize() > 0:
            try:
                self.warden.__event__.wait()
                while self.warden.__event__.is_set() and self.warden.__queue__.qsize() > 0:
                    try:
                        message = self.warden.__queue__.get()
                        if first_zero:
                            if message[0] == 0:
                                if rows != self.__size__ - 1:
                                    print("\rDatos corruptos, continuando . . .")
                                rows = 0
                            else:
                                rows = rows + 1
                            self.data = np.vstack([self.data, message])
                        elif int(message[0]) == 0:
                            self.data = message
                            first_zero = True
                            rows = 0
                    except Exception as e:
                        if first_zero:
                            self.data = self.data[: -(rows + 1)]
                            rows = 0
                        print(e)
                        traceback.print_exc()
                exc = len(self.data) % self.__size__
                if exc > 0:
                    rest = self.data[-exc:]
                    self.data = self.data[:-exc]
                else:
                    rest = np.array([])
                    first_zero = False
                if len(self.data) > 0:
                    self.eeg.add(self.data)
                self.data = rest
                rows = len(self.data) - 1
            except Exception as e:
                print(e)
                self.data = np.array([])
                first_zero = False
                rows = 0
                traceback.print_exc()
                self.eeg.reset()
        self.process_eeg()
        self.warden.unlock_internal_process()
        print("\r\nConsumidor termina")

    def process_eeg(self):
        print("Datos recopilados: " + str(self.eeg.count()))
        usr_mg = managers.get_store_manager()()
        usr_id = self.warden.get_active_user()
        cmd_id = self.warden.get_active_command()
        usr = usr_mg.get_user(usr_id)
        cmd = usr.get_command(cmd_id)

        if self.eeg.is_full():
            print("Procesando datos EEG (" + str(self.eeg.count()) + ")")
            total = self.run_eeg()
            print("Guardando datos EEG")
            cmd.set_eeg(total)
            usr_mg.save_command(usr_id, cmd)
            usr_mg.save()
            if cmd.get_id() != 1:
                self.warden.train()
            self.eeg.reset()
        else:
            usr.remove_command(cmd)
            usr_mg.save()
            print("No hay suficientes datos para EEG")

    def stop(self):
        """
        Stop and cleans the process, it waits until all data be processed.
        """
        print("Terminando . . .")
        self.__stop__.value = True
        self.warden.__event__.set()
        print("Limpiando . . .")
        if self.warden.is_train_mode():
            while self.warden.__queue__.qsize() > 0:
                sys.stdout.write(
                    "Recopilando datos...\rQuedan: " + str(self.warden.__queue__.qsize()) + " elementos por recopilar.")
                sys.stdout.flush()
            print("\rEsperando a EEG . . .")
            self.warden.wait_internal()
        self.warden.unlock_external_process()

    def run_eeg(self):
        """
        Computes all EEG information.
        :return: EEG information
        """
        total = None
        for value in self.eeg:
            data = np.array(
                (value.get_delta(),
                 value.get_theta(),
                 value.get_alpha(),
                 value.get_beta(),
                 value.get_gamma(),
                 value.get_delta_avg(),
                 value.get_theta_avg(),
                 value.get_alpha_avg(),
                 value.get_beta_avg(),
                 value.get_gamma_avg(),
                 value.get_engagement(),
                 value.get_hjorth_complexity(),
                 value.get_hjorth_mobility(),
                 value.get_hjorth_activity(),
                 value.get_pfd(),
                 value.get_kfd(),
                 value.get_hfd())
            )
            if total is None:
                total = data
            else:
                total = np.vstack((total, data))
        return total
