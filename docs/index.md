---
hide:
  - navigation
  - toc
---


<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "SoftwareApplication",
  "name": "Algorithm Selection Framework",
  "description": "A powerful library for algorithm selection and performance prediction",
  "applicationCategory": "DeveloperApplication",
  "operatingSystem": "Any",
}
</script>


# Algorithm Selection Framework (ASF)

ASF is a lightweight yet powerful Python library for algorithm selection and empirical performance prediction. 
It implements various algorithm selection methods, along with algorithm pre-selection, pre-solving schedules and more features to easily create algorithm selection pipeline.
ASF is a modular framework that allows easy extensions to tailor made an algorithm selector for every use-case.
While ASF includes several built-in machine learning models through scikit-learn and XGBoost, it supports every model that complies with the scikit-learn API.
ASF also implements empirical performance prediction, allowing to use different performance scalings.

ASF is written in Python 3 and is intended to use with Python 3.10+. It requires only scikit-learn, NumPy as Pandas as basic requirements. More advanced features (such as hyperparameter optimisation) requires additional dependencies. 


# Cite Us

If you use ASF, please cite the Zenodo DOI. We are currently working on publishing a paper on ASF, but by then a Zenodo citation will do it. 

```bibtex
@software{ASF,
	author = {Hadar Shavit and Holger Hoos},
	doi = {10.5281/zenodo.15288151},
	title = {ASF: Algorithm Selection Framework},
	url = {https://doi.org/10.5281/zenodo.15288151},
	year = 2025,
}
```