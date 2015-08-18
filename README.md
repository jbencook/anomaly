# Anomaly

![build](https://travis-ci.org/find-io/anomaly.svg?branch=master)
[![coverage](https://coveralls.io/repos/find-io/anomaly/badge.svg?branch=master&service=github)](https://coveralls.io/github/find-io/anomaly?branch=master)

A python implementation of [anomaly_detection](https://github.com/pollo/anomaly_detection).

Any anomaly detection class will need to inherit from the `AnomalyDetection`.
Implement `build_model()` and `reconstruct_signal()`.

To run the examples:

1. `pip install -r examples/requirements.txt`
2. `python setup.py install`
3. `python examples/ekg.py examples/a02.dat`
