from ConfigSpace import ConfigurationSpace, Configuration


class Configurable:
    """
    A class that allows for the configuration of various parameters and settings.
    """

    @staticmethod
    def get_configuration_space(
        cs: ConfigurationSpace = None,
        **kwargs,
    ):
        raise NotImplementedError(
            "get_configuration_space() is not implemented for this class"
        )

    @staticmethod
    def get_from_configuration(configuration: Configuration):
        """
        Get the object from the configuration.

        Returns
        -------
        Configurable
            The object.
        """
        raise NotImplementedError(
            "get_from_configuration() is not implemented for this class"
        )
