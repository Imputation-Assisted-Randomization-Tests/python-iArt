from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="causal-i-art",
    version="0.1.0",
    author="Siyu Heng, Jiawei Zhang, and Yang Feng",
    author_email="",
    description="I-ART: A Generalized Framework for Imputation-Assisted Randomization Tests",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jiawei-zhang-a/I-ART",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    python_requires=">=3.8",
    install_requires=[
        'mv-laplace',
        'pandas',
        'numpy',
        'scikit-learn',
        'xgboost',
        'statsmodels',
        'lightgbm',
    ],
)
