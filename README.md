# SYNTHETIC CONTROL USING CAUSAL PY

## data_reader.py
Loads data form PEDIDOS_DET ... and stores time series
in a python dictionary named ts_dict.py

## diana_sc.py
Using ts_dict generates the dataframes for sellers M32 - M42, measuring the
efect on M36 and M40 given a mode passed as a flag performs synthetic control

## command
python diana_sc.py --mode raw

## Flags
* raw: original data
* sd: seasonal decompose smoothing
* ds: double exponential damped smoothing