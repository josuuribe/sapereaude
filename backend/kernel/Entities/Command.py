class Command:
    """
    Class that represents a command entity that stores all functionality related to
    save and execute a command
    """

    def __init__(self):
        """
        Constructor that initializes entity
        """
        self.__name__ = ""
        self.__action__ = ""
        self.__eeg__ = None
        self.__id__ = -1
        self.__parameters__ = None

    @staticmethod
    def version():
        return 1.0

    def get_id(self):
        """
        Get an id
        :return: Id related to this entity
        """
        return self.__id__

    def set_id(self, id):
        """
        Set an id for this entity
        :param id: Id to set
        """
        self.__id__ = id

    def set_name(self, name):
        """
        Set a friendly name for this command
        :param name: Friendly name
        """
        self.__name__ = name

    def get_name(self):
        """
        Get this command name
        :return: Name for this command
        """
        return self.__name__

    def get_action(self):
        """
        Get action for this command, it means the process that will be executed.
        :return: The command that will be executed.
        """
        return self.__action__

    def set_action(self, action):
        """
        Set the action that will be executed when be infered, like an executable.
        :param action: Application to be executed or other command.
        """
        self.__action__ = action

    def get_eeg(self):
        """
        Returns all EEG information related to this command.
        :return: EEG information, usually implemented like a numpy array.
        """
        return self.__eeg__

    def set_eeg(self, eeg):
        """
        Set all EEG information.
        :param eeg: EEG inforamtion to set like a numpy array.
        :return:
        """
        self.__eeg__ = eeg

    def get_parameters(self):
        """
        Get all parameters that will be used to execute an application.
        :return: Parameters to be executed.
        """
        return self.__parameters__

    def set_parameters(self, parameters):
        """
        Set all parameters that will be used to execute an application.
        :param parameters: Parameters to be used while executing an application.
        """
        self.__parameters__ = parameters

    def is_neutral(self):
        """
        Tells if the actual command is the neutral command, it means a command that does not do nothing.
        :return: True if is neutral, False otherwise.
        """
        return self.__id__ == 1
