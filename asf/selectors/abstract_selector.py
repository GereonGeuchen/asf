import pandas as pd
from asf.scenario.scenario_metadata import SelectionScenarioMetadata
from asf.selectors.feature_generator import (
    AbstractFeatureGenerator,
)
from ConfigSpace import ConfigurationSpace, Categorical, Configuration


class AbstractSelector:
    def __init__(
        self,
        metadata: SelectionScenarioMetadata,
        hierarchical_generator: AbstractFeatureGenerator = None,
    ):
        self.metadata = metadata
        self.hierarchical_generator = hierarchical_generator
        self.algorithm_features = None

    def fit(
        self,
        features: pd.DataFrame,
        performance: pd.DataFrame,
        algorithm_features: pd.DataFrame = None,
        **kwargs,
    ):
        if self.hierarchical_generator is not None:
            self.hierarchical_generator.fit(features, performance, algorithm_features)
            features = pd.concat(
                [features, self.hierarchical_generator.generate_features(features)],
                axis=1,
            )
        self.algorithm_features = algorithm_features
        self._fit(features, performance, **kwargs)

    def predict(self, features: pd.DataFrame) -> dict[str, list[tuple[str, float]]]:
        if self.hierarchical_generator is not None:
            features = pd.concat(
                [features, self.hierarchical_generator.generate_features(features)],
                axis=1,
            )
        return self._predict(features)

    def save(self, path):
        pass

    def load(self, path):
        pass

    def get_configuration_space(cs: ConfigurationSpace = None, **kwargs):
        """
        Get the configuration space for the selector.

        Parameters
        ----------
        cs : ConfigurationSpace, optional
            The configuration space to use. If None, a new one will be created.

        Returns
        -------
        ConfigurationSpace
            The configuration space for the selector.
        """
        raise NotImplementedError(
            "get_configuration_space() is not implemented for this selector"
        )

    @staticmethod
    def get_from_configuration(configuration: Configuration):
        """
        Get the configuration space for the selector.

        Returns
        -------
        AbstractSelector
            The selector.
        """
        raise NotImplementedError(
            "get_from_configuration() is not implemented for this selector"
        )

    @staticmethod
    def _add_hierarchical_generator_space(
        cs,
        hierarchical_generator: list[AbstractFeatureGenerator] | None = None,
        **kwargs,
    ):
        """
        Add the hierarchical generator space to the configuration space.

        Parameters
        ----------
        cs : ConfigurationSpace
            The configuration space to use.
        hierarchical_generator : list[AbstractFeatureGenerator] | None, optional
            The list of hierarchical generators to add. Defaults to None.
        **kwargs : dict
            Additional keyword arguments to pass to the model class.
        """
        if hierarchical_generator is not None:
            if "hierarchical_generator" in cs:
                return

            cs.add(
                Categorical(
                    name="hierarchical_generator",
                    choices=hierarchical_generator,
                )
            )

            for generator in hierarchical_generator:
                generator.get_configuration_space(cs=cs, **kwargs)

        return cs
