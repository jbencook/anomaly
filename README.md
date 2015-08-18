# Anomaly

A python implementation of [anomaly_detection](https://github.com/pollo/anomaly_detection).

Any anomaly detection system will need to inherit from the `AnomalyDetection`.
Implement `build_model()` and `reconstruct_signal()`.

To run the examples:

1. `pip install -r examples/requirements.txt`
2. `python setup.py install`
3. `python examples/ekg.py examples/a02.dat`
