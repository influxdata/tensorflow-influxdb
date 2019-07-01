# Introduction to InfluxDB-Tensorflow

By the end of this presentation participants should be able to...

* Describe what InfluxDB is
* Describe what TensorFlow is
* Explain what Data Engineering is and why it is distinct from Data Science
* Explain how InfluxDB can be used with TensorFlow to make the data engineering problem easier
* Understand how InfluxDB models time-series
* Query data from InfluxDB
* Use the presented example of InfluxDB-Tensorflow together
* Express the nicety of being able to transition from historical to real-time data, within the same system

## Environment

### Conda

Create a conda environment with tensorflow and scikit-learn.

```sh
 $ conda create --name tf3 --file tf3.txt
 $ conda activate tf3
```

### InfluxDB client

InfluxDB does not have a conda environment so we need to install with with pip.

```sh
 $ pip install influxdb
```

### Load Weather Data into InfluxDB
The data for this course can be found in line protocol in the data directory.

To import this weather data into a running InfluxDB run:

```sh
 $ gunzip data/201701hourly.lp.gz
 $ influx -import -path="data/201701hourly.lp" -precision s
```

#### Schema
Database name is `tensorflowdb`

**Example**
```txt
qclcd,wban=00102 wet_bulb_temp=27.0 1451649600
```

* The tag `wban` is the ID for the wireless body area network.
* The field `wet_bulb_temp` is the temperature in farenheit.


## Notebook

The tensorflux.ipynb contains an example time-series prediction of weather using
data from influxdb.
