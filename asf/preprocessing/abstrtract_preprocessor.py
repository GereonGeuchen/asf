class AbstractPreprocessor:
    """
    An abstract base class for data preprocessing.

    This class defines the interface for preprocessing steps, including
    methods for fitting to data and transforming data. Subclasses should
    implement the `fit` and `transform` methods to define specific
    preprocessing behavior.

    Methods:
        fit(data): Fits the preprocessor to the provided data.
        transform(data): Transforms the provided data using the fitted preprocessor.
    """

    def __init__(self):
        """
        Initializes the AbstractPreprocessor. This constructor does not
        perform any specific initialization and is meant to be overridden
        by subclasses if needed.
        """
        pass

    def fit(self, data: any) -> None:
        """
        Fits the preprocessor to the provided data.

        Args:
            data (any): The data to fit the preprocessor to. The type of
                        data depends on the specific implementation.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def transform(self, data: any) -> any:
        """
        Transforms the provided data using the fitted preprocessor.

        Args:
            data (any): The data to transform. The type of data depends on
                        the specific implementation.

        Returns:
            any: The transformed data. The type of the returned data depends
                 on the specific implementation.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError
