# Algorithm Selection Framework (ASF)

ASF is a powerful library for algorithm selection and performance prediction. It allows users to easily create and use algorithm selectors with minimal code.

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

## Documentation

For detailed documentation and examples, please refer to the official documentation.

## Contributing

We welcome contributions! Please see our contributing guidelines for more details.

## License

ASF is licensed under the MIT License. See the LICENSE file for more details.