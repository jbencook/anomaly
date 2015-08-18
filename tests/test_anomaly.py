import anomaly
import numpy as np
import unittest


class DummyAnomaly(anomaly.TimeSeriesAnomalyDetection):
    def __init__(self):
        pass


class SinAnomaly(anomaly.TimeSeriesAnomalyDetection):
    def build_model(self, data):
        pass

    def reconstruct_signal(self, data):
        return np.sin(np.linspace(-10, 10, 1000))[:, np.newaxis]


class TestAnomaly(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.x = np.sin(np.linspace(-10, 10, 1000))
        cls.anomaly_i = 250
        cls.x[cls.anomaly_i] = 1000.

    def test_not_implemented_errors(self):
        dummy = DummyAnomaly()
        with self.assertRaises(NotImplementedError):
            dummy.build_model(self.x)

        with self.assertRaises(NotImplementedError):
            dummy.reconstruct_signal(self.x)

    def test_anomaly_detection(self):
        sin_model = SinAnomaly()
        sin_model.build_model(self.x)
        anomalies = sin_model.detect_anomalies(self.x, .01)

        self.assertEqual(anomalies[0].index, self.anomaly_i)


