class IConsumer:
    """
    This interface must be implemented for consume information.
    """

    def start(self):
        """
        Start consumer.
        """
        raise NotImplementedError

    def lock(self):
        """
        Lock producer component.
        """
        raise NotImplementedError

    def stop(self):
        """
        Stop consumer
        """
        raise NotImplementedError

    @staticmethod
    def version():
        """
        Returns actual version.
        :return: Actual version.
        """
        return 1.0
