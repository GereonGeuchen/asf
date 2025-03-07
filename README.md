[![PyPI - Version](https://img.shields.io/pypi/v/asf-lib)](https://pypi.org/project/asf-lib/)
[![Python versions](https://img.shields.io/pypi/pyversions/asf-lib)](https://pypi.org/project/asf-lib/)
[![License](https://img.shields.io/pypi/l/asf-lib?color=informational)](LICENSE)
[![Python application](https://github.com/hadarshavit/asf/actions/workflows/python-app.yml/badge.svg)](https://github.com/hadarshavit/asf/actions/workflows/python-app.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14957286.svg)](https://doi.org/10.5281/zenodo.14957286)

# ASF: Algorithm Selection Framework

ASF is a powerful library for algorithm selection and performance prediction. It allows users to easily create and use algorithm selectors with minimal code.

> NOTE: ASF is still under construction (early alpha). Therefore, not only the API can change, but there might be some bugs in the implementations of the selectors. For the common methods (multi class classification, pairwise regression / classification as well as simple ranking) we checked the performance and the implementation and they can be safely used. We will release in the near future a benchmark of all methods on ASlib scenarios which will validate the performance.

## Features

- Easy-to-use API for creating algorithm selectors
- Supports various selection models including pairwise classifiers, multi-class classifiers, and performance models
- Integration with popular machine learning libraries like scikit-learn

## Quick Start

You can create an algorithm selector with just 2 lines of code. Here is an example using the `PairwiseClassifier`:

```python
from asf.selectors import PairwiseClassifier
from sklearn.ensemble import RandomForestClassifier

# Create a PairwiseClassifier
selector = PairwiseClassifier(model_class=RandomForestClassifier, metadata=your_metadata)

# Fit the selector with feature and performance data
selector.fit(dummy_features, dummy_performance)

# Predict the best algorithm for new instances
predictions = selector.predict(new_features)
```

## Future Features

In the future, ASF will include more features such as:

- Empirical performance prediction
- Feature selection
- Support for ASlib scenarios
- And more!

## Installation

To install ASF, use pip:
```python
pip install asf-lib
```

## Documentation

For detailed documentation and examples, please refer to the official documentation.

## Contributing

We welcome contributions! Please see our contributing guidelines for more details.

## License

ASF is licensed under the MIT License. See the LICENSE file for more details.
