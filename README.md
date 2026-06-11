### HydroEcoLSTM_Lite 


- HydroEcoLSTM_Lite is a Lite version of the HydroEcoLSTM package.
- Model performance for > 1500 CAMELS-DE catchment for test period (2011-2020) median NSE = 0.853 
- TODO: Train on GPU (currently only train on CPU is possible)



**Citation**

**Nguyen, T.V.**, Tran, V.N., Tran, H., Binh, D.V., Duong, T.D., Dang, T.D., Ebeling, P. (2025). *HydroEcoLSTM*: A Python package with graphical user interface for hydro-ecological modeling with long short-term memory neural network. Ecological Informatics, 102994. [10.1016/j.ecoinf.2025.102994](https://doi.org/10.1016/j.ecoinf.2025.102994).

### Quick start

Installation with Anaconda using environment file following the steps listed below. 

```python
# 1. Download the environment file from https://github.com/tamnva/HydroEcoLSTM_Lite/tree/master/environment/environment.yml file

# 2. Create virtual environment from that environment file using anaconda
conda env create -f environment.yml
conda activate flowstats_env

# 2. Or you can install in your exising environment
pip install https://github.com/tamnva/HydroEcoLSTM_Lite/archive/refs/heads/master.zip

# 3. Run the example in this HydroEcoLSTM/examples/camels_de or HydroEcoLSTM/examples/camels_ch
#    "main.py":         python file for running this example
#    'config.yml'       configuration file (please adjust it)
#    'time_series.csv'  contains the time series and statics inputs to run the model
#                          in the time_series.csv must have two columns "id" and "time"
#                          in the static_attributes.csv must have column "id"
#    Please read the README.md within this folder regarding the orginal data sources and license
```

