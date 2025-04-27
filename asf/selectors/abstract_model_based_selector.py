from asf.selectors.abstract_selector import AbstractSelector
from asf.predictors import SklearnWrapper
from sklearn.base import ClassifierMixin, RegressorMixin
from functools import partial
from typing import Type, Callable, Any, Union
from pathlib import Path
import joblib


class AbstractModelBasedSelector(AbstractSelector):
    """
    An abstract base class for selectors that utilize a machine learning model
    for selection purposes. This class provides functionality to initialize
    with a model class, save the selector to a file, and load it back.

    Attributes:
        model_class (callable): A callable that represents the model class to
            be used. If the provided model_class is a subclass of
            `ClassifierMixin` or `RegressorMixin`, it is wrapped using
            `SklearnWrapper`.

    Methods:
        save(path):
            Saves the current instance of the selector to the specified file
            path using `joblib`.

        load(path):
            Loads a previously saved instance of the selector from the
            specified file path using `joblib`.

    Args:
        model_class (type or callable): The model class to be used. It can be
            a subclass of `ClassifierMixin` or `RegressorMixin`, or any other
            callable model class.
        **kwargs: Additional keyword arguments to be passed to the parent
            `AbstractSelector` class.
    """

    def __init__(self, model_class: Union[Type, Callable], **kwargs: Any) -> None:
        """
        Initializes the AbstractModelBasedSelector.

        Args:
            model_class (Union[Type, Callable]): The model class or a callable
                that returns a model instance. If a scikit-learn compatible
                class is provided, it's wrapped with SklearnWrapper.
            **kwargs (Any): Additional keyword arguments passed to the
                parent class initializer.
        """
        super().__init__(**kwargs)

        if isinstance(model_class, type) and issubclass(
            model_class, (ClassifierMixin, RegressorMixin)
        ):
            self.model_class: Callable = partial(SklearnWrapper, model_class)
        else:
            self.model_class: Callable = model_class

    def save(self, path: Union[str, Path]) -> None:
        """
        Saves the selector instance to a file using joblib.

        Args:
            path (Union[str, Path]): The file path where the selector
                should be saved.
        """
        joblib.dump(self, path)

    @staticmethod
    def load(path: Union[str, Path]) -> "AbstractModelBasedSelector":
        """
        Loads a selector instance from a file using joblib.

        Args:
            path (Union[str, Path]): The file path from which to load the
                selector.

        Returns:
            AbstractModelBasedSelector: The loaded selector instance.
        """
        return joblib.load(path)
