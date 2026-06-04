### HydroEcoLSTM_Lite 


- HydroEcoLSTM_Lite is a Lite version of the HydroEcoLSTM package. 
- TODO: Train on GPU (currently only train on CPU is possible)



**Citation**

**Nguyen, T.V.**, Tran, V.N., Tran, H., Binh, D.V., Duong, T.D., Dang, T.D., Ebeling, P. (2025). *HydroEcoLSTM*: A Python package with graphical user interface for hydro-ecological modeling with long short-term memory neural network. Ecological Informatics, 102994. [10.1016/j.ecoinf.2025.102994](https://doi.org/10.1016/j.ecoinf.2025.102994).

### Quick start

Installation with Anaconda using environment file following the steps listed below. 

```python
# 1. Download the environment file from https://github.com/tamnva/HydroEcoLSTM_Lite/tree/master/environment/environment.yml file

# 2. Create virtual environment from that environment file using anaconda
conda env create -f environment.yml
conda activate hydroecolstm_lite

# 2. Install the lastest version from github
pip install git+https://github.com/tamnva/HydroEcoLSTM_Lite.git

# 3. Run the example in this HydroEcoLSTM/examples/1_streamflow_simulation
#    "main.py":     python file for running this example
#    'config.yml'   configuration file (please adjust it)
#    'data'         contains the time series and statics inputs to run the model
#                   in the time_series.csv must have two columns "id" and "time"
#                   in the static_attributes.csv must have column "id"
```

