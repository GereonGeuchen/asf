---
hide:
  - navigation
  - toc
---

# Installation

ASF is written in Python3 and requires Python version 3.10+.
The basic installation is lightweight and requires only NumPy, Pandas and scikit-learn.

ASF is currently tested on Linux machines. Mac and Windows (official) support will be released in the near future.

To install the base version run 
```bash
pip install asf-lib
```

## Additional options

Additional options include:

- XGBoost model suppot `pip install asf-lib[xgb]`
- PyTorch-based models `pip install asf-lib[nn]`
- ASlib scenarios reading `pip install asf-lib[aslib]`
