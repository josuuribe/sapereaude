class IProducer:
    """
    This interface must be implemented for produce information.
    """

    def start(self):
        """
        Start producer
        """
        raise NotImplementedError

    def stop(self):
        """
        Stop producer
        :return:
        """
        raise NotImplementedError

    @staticmethod
    def version():
        """
        Returns actual version.
        :return: Actual version.
        """
        return 1.0
