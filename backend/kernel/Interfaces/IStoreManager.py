class IStoreManager:
    """
    Interfaz to store information.
    """

    def __init__(self):
        self.__users__ = list()

    @staticmethod
    def version():
        """
        Returns actual version.
        :return: Actual version.
        """
        return 1.0

    def find_user(self, user_id):
        """
        Finds a user id given.
        :param user_id: User id to look for.
        :return: User found or None if not found.
        """
        user = (x for x in self.__users__ if x.get_id() == user_id)
        return user

    def get_user(self, user_id):
        """
        Get user id given.
        :param user_id: User id to look for.
        :return: User found or None if not found.
        """
        raise NotImplementedError

    def get_users(self):
        """
        Get all users.
        :return: All users.
        """
        raise NotImplementedError

    def add_user(self, user):
        """
        Add an user id given.
        :param user: User to add.
        """
        raise NotImplementedError

    def remove_user(self, user):
        """
        Remove a user id given.
        :param user: User to remove.
        """
        raise NotImplementedError

    def update_user(self, user):
        """
        Update a user
        :param user: User to update
        """
        raise NotImplementedError

    def get_model_folder(self, user_id):
        """
        Get the model folder, the folder where neural models will be stored.
        :param user_id: User id associated to folders.
        :return: Model folder path.
        """
        raise NotImplementedError

    def save_command(self, user_id, command):
        """
        Saves a command, this means save EEG information for a command.
        :param user_id: User id to get command.
        :param command: User command to be saved.
        """
        raise NotImplementedError

    def load_command(self, user_id, command_id):
        """
        Loads a command, this means fill command with all EEG information.
        :param user_id: User id to load command.
        :param command_id: User command to load.
        :return:
        """
        raise NotImplementedError

    def refresh(self):
        """
        Refresh all information for this user, very useful for multiprocessing.
        """
        raise NotImplementedError

    def save(self):
        """
        Saves all information updated during the session.
        """
        raise NotImplementedError
