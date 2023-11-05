# I-ART: Imputation-Assisted Randomization Tests

## Description

I-ART (Imputation-Assisted Randomization Tests) is a Python package designed for conducting finite-population-exact randomization tests in design-based causal studies with missing outcomes. It offers a robust solution to handle missing data in causal inference, leveraging the potential outcomes framework and integrating various outcome imputation algorithms.

## Installation

To install I-ART, run the following command:

```bash
pip install causal-i-art
```

## Usage

Here is a basic example of how to use I-ART:

```python
import numpy as np
from i_art import iartest

Z = np.array([1, 1, 1, 1, 0, 0, 0, 0])
X = np.array([[5.1, 3.5], [4.9, np.nan], [4.7, 3.2], [4.5, np.nan], [7.2, 2.3], [8.6, 3.1], [6.0, 3.6], [8.4, 3.9]])
Y = np.array([[4.4, 0.5], [4.3, 0.7], [4.1, np.nan], [5.0, 0.4], [1.7, 0.1], [np.nan, 0.2], [1.4, np.nan], [1.7, 0.4]])
result = iartest(Z=Z, X=X, Y=Y, L=1000, verbose=True)
```

## Features

- Conducts finite-population-exact randomization tests.
- Handles missing data in causal inference studies.
- Supports various outcome imputation algorithms.
- Offers covariate adjustment in exact randomization tests.


## Contributing

Your contributions to I-ART are highly appreciated! If you're looking to contribute, we encourage you to open issues for any bugs or feature suggestions, or submit pull requests with your proposed changes. 

### Setting Up a Development Environment

To set up a development environment for contributing to I-ART, follow these steps:

```bash
python -m venv venv
source venv/bin/activate 
pip install -r requirements.txt
```
This creates a virtual environment (`venv`) for Python and activates it, allowing you to work on the package without affecting your global Python environment.

## License
This project is licensed under the MIT License

## Citation
If you use I-ART in your research, please consider citing it:

```code
@misc{heng2023designbased,
      title={Design-Based Causal Inference with Missing Outcomes: Missingness Mechanisms, Imputation-Assisted Randomization Tests, and Covariate Adjustment}, 
      author={Siyu Heng and Jiawei Zhang and Yang Feng},
      year={2023},
      eprint={2310.18556},
      archivePrefix={arXiv},
      primaryClass={stat.ME}
}
```
