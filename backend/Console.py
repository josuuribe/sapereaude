from kernel.Entities.Command import Command
from kernel.Entities.User import User
from kernel.Warden import Warden
import managers
import sys
import threading
import time
import datetime


class Console:
    """
    Simple class that acts like UI in console mode
    """

    def __init__(self, args):
        self.args = args
        self.usr_manager_obj = managers.get_store_manager()
        self.usr_manager = self.usr_manager_obj()
        self.seconds = 0
        self.message = ""
        self.__timer__ = WardenTimer()

    def list_training(self, user):
        trainings = user.get_commands()
        training = None
        if len(trainings) > 0:
            for training in trainings:
                print(str(training.get_id()) + ") " + training.get_name())
            training_id = input("Inserte codigo: ").lower()
            training = user.get_command(int(training_id))
            if training is None:
                print("Entrenamiento no encontrado")
                self.list_training(user)
        return training

    def training_menu_detail(self, user, command):
        command_name = input("Inserte nombre: ").lower()
        command.set_name(command_name)
        command_action = input("Inserte accion: ").lower()
        command.set_action(command_action)
        parameters = list()
        parameter = "go"
        while parameter is not '':
            parameter = input("Introduzca parámetro o nada para salir.")
            parameters.append(parameter)

        command.set_parameters(parameters)

        yes_no = "*"
        while yes_no not in ('s', 'n'):
            yes_no = input("Indique si quiere comenzar el entrenamiento (s/n):").lower()
        if yes_no == 's':
            user.add_command(command)
            self.usr_manager.save()
            self.__timer__.set_active_user(user.get_id())
            self.__timer__.set_active_command(command.get_id())
            self.__timer__.set_train_mode()
            self.__timer__.start()

            self.__timer__.lock()
            self.__timer__.wait()
            print("\rEntrenamiento completado.")
        self.training_menu(user)

    def training_menu(self, user):
        self.__timer__.set_active_user(user.get_id())
        training_input = input("Entrenamiento: Agregar / Modificar / Borrar / Salir (A/M/B/S)").lower()
        command = Command()
        if training_input == 'a':
            if len(user.get_commands()) == 0:
                input(
                    "No existe ninún comando.\nPrimero hay que establecer la línea neutra, el entrenamiento durará 15 segundos.\nPresione una tecla cuando esté preparado.")
                command.set_name("Neutral")
                user.add_command(command)
                self.usr_manager.save()
                self.__timer__.set_active_user(user.get_id())
                self.__timer__.set_active_command(command.get_id())
                self.__timer__.set_train_mode()
                self.__timer__.start()

                self.__timer__.lock()
                self.__timer__.wait()

                print("\r Comando terminado.")
            else:
                self.training_menu_detail(user, command)
            self.training_menu(user)
        elif training_input == 'm':
            command = self.list_training(user)
            if command is None:
                print("Entrenamiento no encontrado.")
                self.training_menu(user)
            else:
                self.training_menu_detail(user, command)
                user.update_command(command)
        elif training_input == 'b':
            command = self.list_training(user)
            if command is None:
                print("Entrenamiento no encontrado.")
                self.training_menu(user)
            else:
                user.remove_command(command)
        elif training_input == 's':
            self.show()

    def user_menu(self, user):
        user_code = input("Inserte codigo: ").lower()
        user.set_code(user_code)
        user_name = input("Inserte nombre de usuario: ").lower()
        user.set_name(user_name)
        return user

    def list_users(self):
        users = self.usr_manager.get_users()
        user = None
        if len(users) == 0:
            print("No hay usuarios")
        if len(users) > 0:
            for user in users:
                print(str(user.get_id()) + ") " + user.get_name())
            user_id = input("Inserte código a buscar: ").lower()
            user = self.usr_manager.get_user(user_id)
            if user is None:
                print("Usuario no encontrado")
                self.list_users()
        return user

    def show(self):
        input_menu = ""
        while input_menu != 's':
            input_menu = input("Usuario / Entrenamiento / Inferencia / Guardar / Salir (u/e/i/g/s)").lower()
            if input_menu == 'u':
                input_menu = input("Usuario: Agregar / Modificar / Borrar (a/m/b)").lower()
                user = User()
                if input_menu == 'a':
                    user = self.user_menu(user)
                    self.usr_manager.add_user(user)
                    self.training_menu(user)
                elif input_menu == 'm':
                    user = self.list_users()
                    if user is not None:
                        self.user_menu(user)
                        self.usr_manager.update_user(user)
                        self.training_menu(user)
                elif input_menu == 'b':
                    user = self.list_users()
                    if user is not None:
                        self.usr_manager.remove_user(user)
            elif input_menu == 'e':
                user = self.list_users()
                if user is not None:
                    self.training_menu(user)
            elif input_menu == 'i':
                yes_no = "*"
                while yes_no not in ('s', 'n'):
                    yes_no = input("Indique si quiere comenzar la inferencia (s/n):").lower()
                if yes_no == 's':
                    user = self.list_users()
                    if user is not None:
                        self.__timer__.set_active_user(user.get_id())
                        self.__timer__.set_inference_mode()
                        self.__timer__.start()

                        self.__timer__.lock()
                        self.__timer__.wait()
                        print("Inferencia completada")
                self.show()
            elif input_menu == 'g':
                self.usr_manager.save()
                print("Fichero guardado")
            elif input_menu == 's':
                print("Saliendo . . .")
            else:
                print("Opción no encontrada")


class WardenTimer():
    warden = None

    def __init__(self):
        self.x = 0
        self.message = ""
        consumer = managers.get_consumer_manager()
        producer = managers.get_producer_manager()
        store_manager = managers.get_store_manager()
        WardenTimer.warden = Warden(producer, consumer, store_manager)

    def wait(self):
        self.warden.wait_process()

    def lock(self):
        self.warden.lock_external_process()

    def unlock(self):
        self.warden.unlock_external_process()

    def show_message(self):
        sys.stdout.write("\r" + self.message + str(self.x))
        sys.stdout.flush()

    def timer_message(self):
        while self.x > 0:
            timer = threading.Timer(1.0, self.show_message)
            self.x -= 1
            timer.start()
            time.sleep(1.5)
        sys.stdout.write("\r")

    def set_active_command(self, command_id):
        self.warden.set_active_command(command_id)

    def set_active_user(self, user_id):
        self.warden.set_active_user(user_id)

    def get_active_user(self):
        return self.warden.get_active_user()

    def set_train_mode(self):
        self.warden.set_train_mode()

    def set_inference_mode(self):
        self.warden.set_inference_mode()

    def get_active_command(self):
        return self.warden.get_active_command()

    def stop(self):
        print("Parando Warden -> {}.".format(datetime.datetime.now().time()))
        WardenTimer.warden.stop()
        print("\rWarden parado")

    def start(self):
        if self.warden.is_train_mode():
            print("Arrancando modo entrenamiento -> {}.".format(datetime.datetime.now().time()))
            duration = managers.get_duration()
            print("Arrancando a Warden durante " + str(duration) + " segundos")
            self.message = "Empezamos en: "
            self.x = 3
            self.timer_message()
            timer = threading.Timer(duration, self.stop)
            timer.start()
            WardenTimer.warden.start()
        elif self.warden.is_inference_mode():
            print("Arrancando modo inferencia -> {}.".format(datetime.datetime.now().time()))
            WardenTimer.warden.start()
