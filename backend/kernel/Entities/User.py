from kernel.Entities import Command


class User:
    """
    Class that represents a user entity that stores all functionality related to
    application user.
    """

    def __init__(self):
        self.__id__ = None
        self.__name__ = None
        self.__code__ = None
        self.__commands__ = list()
        self.__ml__ = None

    def get_id(self):
        """
        Get user id.
        :return: The user id.
        """
        return self.__id__

    def set_id(self, id):
        """
        Set user id.
        :param id: The user id
        """
        self.__id__ = id

    def get_name(self):
        """
        Get user name.
        :return: The user name.
        """
        return self.__name__

    def set_name(self, name):
        """
        Set user name.
        :param name: User name to set.
        """
        self.__name__ = name

    def get_code(self):
        """
        Get user code.
        :return: The user code.
        """
        return self.__code__

    def set_code(self, code):
        """
        Set user code, this is a special code for this user, like a login.
        :param code: User code, generally alphanumeric.
        """
        self.__code__ = code

    def get_commands(self):
        """
        Get all commands related to this user.
        :return: All command for this user.
        """
        return self.__commands__

    def get_command(self, id):
        """
        Returns a command id given.
        :param id: Command id to look for.
        :return: The command found.
        """
        for command in self.get_commands():
            if command.get_id() == id:
                return command
        return None

    def add_command(self, command: Command):
        """
        Adds a command to this user.
        :param command: The command to add.
        """
        max_id = 0
        if len(self.__commands__) > 0:
            max_id = max(self.__commands__, key=lambda t: t.get_id()).get_id()
        command.set_id(max_id + 1)
        self.__commands__.append(command)

    def update_command(self, command):
        """
        Updates a command, the command will be replaced.
        :param command: The command to update
        """
        index, _ = (x for i, x in enumerate(self.get_users()) if x.get_id() == command.get_id())
        self.__commands__[index] = command

    def remove_command(self, command: Command.Command):
        """
        Removes a command, the command will be removed from this user, it is based on an id search.
        :param command: Command to remove.
        :return:
        """
        old_command = self.get_command(command.get_id())
        if old_command is not None:
            self.__commands__.remove(old_command)
