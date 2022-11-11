# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
The goal of the project is to implement best coding practices on the provided churn_notebook.ipynb. 
The notebook contains the solution to identify credit card customers that are most likely to churn, but without 
implementing the engineering and software best practices.

## Files and data description
The root directory contains:

`Guide.ipynb`                        Given: Getting started and troubleshooting tips

`churn_notebook.ipynb`               Given: Contains the code to be refactored

`churn_library.py`                   Implemented: Main functions

`churn_script_logging_and_tests.py`  Implemented tests and logs

`README.md`            

`data`                               folder contains the customer data

`images`                             folder contains EDA and classification results on train/test data

`logs`                               folder contains testing log results

`models`                             folder contains trained models
 
## Running Files
Install mamba[https://pypi.org/project/mamba/].
Create a conda environment:

```bash
conda create -n customer_churn python=3.8
```

Activate the environment:

```bash
mamba activate customer_churn 
```

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the requirements:
```bash
pip install -r requirements.txt
```

To run implemented tests:
```bash
ipython churn_script_logging_and_tests.py 
```

To run the project:
```bash
ipython churn_library.py
```

