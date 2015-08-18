from tdigest import TDigest
import numpy as np

class Anomaly(object):
    def __init__(self, data, error, index):
        self.data = data
        self.error = error
        self.index = index


class TimeSeriesAnomalyDetection(object):
    def build_model(self, data):
        raise NotImplementedError

    def reconstruct_signal(self, data):
        raise NotImplementedError

    def compute_error(self, actual_point, reconstructed_point):
        error = actual_point - reconstructed_point
        return np.linalg.norm(error, 2)

    def detect_anomalies(self, data, anomaly_fraction):
        data = np.asanyarray(data)
        if len(data.shape) == 1:
            data = data[:, np.newaxis]
        signal = self.reconstruct_signal(data)
        digest = TDigest()

        n = data.shape[0]
        delta = np.zeros(data.shape)

        for i in xrange(n):
            error = self.compute_error(data[i, :], signal[i, :])
            delta[i, :] = error
            digest.update(np.abs(error))

        threshold = digest.quantile(1 - anomaly_fraction)

        anomalies = []
        for i in xrange(n):
            element = delta[i]
            if np.abs(element) > threshold:
                anomalies.append(Anomaly(data[i], element, i))

        return anomalies
