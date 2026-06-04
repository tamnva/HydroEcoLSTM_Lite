### HydroEcoLSTM_Lite <a href="https://github.com/tamnva/hydroecolstm/tree/master/docs/images/logo.svg"><img src="docs/images/logo.svg" align="right" height="120" /></a>

[![Documentation Status](https://readthedocs.org/projects/hydroecolstm/badge/?version=latest)](https://hydroecolstm.readthedocs.io/en/latest/?badge=latest) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10673255.svg)](https://doi.org/10.5281/zenodo.10673255) [![PyPI Latest Release](https://img.shields.io/pypi/v/hydroecolstm)](https://pypi.org/project/hydroecolstm/) 


- HydroEcoLSTM_Lite is a Lite version of the HydroEcoLSTM package. 
- TODO: Train on GPU (currently only train on CPU is possible)



**Citation**

**Nguyen, T.V.**, Tran, V.N., Tran, H., Binh, D.V., Duong, T.D., Dang, T.D., Ebeling, P. (2025). *HydroEcoLSTM*: A Python package with graphical user interface for hydro-ecological modeling with long short-term memory neural network. Ecological Informatics, 102994. [10.1016/j.ecoinf.2025.102994](https://doi.org/10.1016/j.ecoinf.2025.102994).

### Quick start

Installation with Anaconda using environment file following the steps listed below. 

```python
# 1. Create the environment from environment.yml file (see link above)
conda env create -f environment.yml
conda activate hydroecolstm_env

# 2. Install the lastest version from github
pip install git+https://github.com/tamnva/HydroEcoLSTM_Lite.git

# 3. Run the example in this HydroEcoLSTM/examples/1_streamflow_simulation
#    "main.py":     python file for running this example
#    'config.yml'   configuration file (please adjust it)
#    'data'         contains the time series and statics inputs to run the model
#                   in the time_series.csv must have two columns "id" and "time"
#                   in the static_attributes.csv must have column "id"
```

