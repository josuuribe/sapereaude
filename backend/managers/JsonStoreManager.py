import kernel
from kernel.Interfaces.IStoreManager import IStoreManager
import jsonpickle
import os
import numpy as np


class JsonStoreManager(IStoreManager):

    def __init__(self):
        """
        Constructor that initializes entity
        """
        super().__init__()
        self.refresh()

    def refresh(self):
        """
        Reloads file with new information, in particular recreates the user collection and all info associated.
        """
        root = kernel.get_connection_string()
        path_to_user = os.path.join(root, "users.json")
        try:
            f = open(path_to_user, "r")
            json = f.read()
            frozen = jsonpickle.decode(json)
            self.__users__ = frozen.__users__
        except Exception as e:
            print("Initial users file not found")
        finally:
            if 'f' in locals() and f is not None:
                f.close()

    @staticmethod
    def version():
        return 1.0

    def get_user(self, user_id):
        try:
            for user_to_search in self.get_users():
                if user_to_search.get_id() == int(user_id):
                    return user_to_search
        except Exception as e:
            return None
        return None

    def get_users(self):
        return self.__users__

    def add_user(self, user):
        max_id = 0
        if len(self.__users__) > 0:
            max_id = max(self.__users__, key=lambda t: t.get_id()).get_id()
        user.set_id(max_id + 1)
        self.__users__.append(user)

    def remove_user(self, user):
        self.__users__.remove(user)

    def update_user(self, user):
        for i, user_to_update in enumerate(self.get_users()):
            if user_to_update.get_id() == user.get_id():
                self.__users__[i] = user

    def get_user_folders(self, user_id):
        """
        Get the user folder and creates them if necessary.
        :param user_id: User id related to these folders.
        :return: 3 folders, the user folder, the command folder to store EEG information and the model folder.
        """
        root = kernel.get_connection_string()
        user = self.get_user(user_id)
        path_to_user = os.path.join(root, str(user.get_id()) + "_" + str(user.get_name()))
        os.makedirs(path_to_user, exist_ok=True)

        path_to_command = os.path.join(path_to_user, "cmd")
        os.makedirs(path_to_command, exist_ok=True)

        path_to_model = os.path.join(path_to_user, "model")
        os.makedirs(path_to_model, exist_ok=True)
        return path_to_user, path_to_command, path_to_model

    def save_command(self, user_id, command):
        """
        Saves command in command folder.
        :param user_id: User id to find user folder.
        :param command: Command folder to save.
        """
        path_to_user, path_to_command, _ = self.get_user_folders(user_id)
        path_data = os.path.join(path_to_command, str(command.get_id()))
        np.savetxt(path_data, command.get_eeg())

    def load_command(self, user_id, command):
        """
        Loads a command using EEG saved file data.
        :param user_id: User id to find user folder.
        :param command: Command folder to save.
        :return:
        """
        _, path_to_command, _ = self.get_user_folders(user_id)
        path_data = os.path.join(path_to_command, str(command.get_id()))
        try:
            data = np.loadtxt(path_data)
            command.set_eeg(data)
        except Exception as e:
            print("No se ha podido cargar el archivo EEG")
            return None
        return command

    def save(self):
        """
        Saves all information in users.json file.
        """
        try:
            frozen = jsonpickle.encode(self)
            root = kernel.get_connection_string()
            path_to_user = os.path.join(root, "users.json")
            f = open(path_to_user, "w+")
            f.write(frozen)
            f.close()
        except Exception:
            print("Error guardando datos")

    def get_model_folder(self, user_id):
        """
        Returns model folder.
        :param user_id: User id to find user folder.
        :return: Model folder.
        """
        _, _, path_to_model = self.get_user_folders(user_id)
        return path_to_model
