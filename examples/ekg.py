import numpy as np
import struct
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from sklearn.cluster import KMeans

from anomaly import TimeSeriesAnomalyDetection


class EKGAnomalyDetection(TimeSeriesAnomalyDetection):
    def __init__(self, anomaly_fraction=.1, window=32, step=2, samples=200000):
        self.anomaly_fraction = anomaly_fraction
        self.window = window
        self.step = step
        self.samples = samples

    def build_model(self, x):
        self.window_vector = np.zeros(self.window)
        for i in xrange(self.window):
            w = np.sin(np.pi * i / (self.window - 1))
            self.window_vector[i] = np.square(w)

        if len(x.shape) == 2:
            x = x[:, 0]

        r = np.zeros((self.samples, self.window))
        for i in xrange(self.samples):
            offset = i * self.step
            row = x[offset:offset+self.window] * self.window_vector
            scale = np.linalg.norm(row)
            r[i, :] = row / scale

        self.model = KMeans(n_clusters=50, max_iter=20)
        self.model.fit(r)

    def reconstruct_signal(self, x):
        if len(x.shape) == 2:
            x = x[:, 0]

        reconstructed_signal = np.zeros(len(x))

        row = np.zeros(self.window)
        row[self.window/2:] = x[:self.window/2]
        scale = np.linalg.norm(row)
        row /= scale
        ndx = self.model.predict(row)[0]
        current = self.model.cluster_centers_[ndx, :]

        reconstructed_signal[:self.window/2] += current[self.window/2:]

        for i in xrange(self.window/2, len(x)-self.window/2, self.window/2):
            row = x[i-self.window/2:i+self.window/2] * self.window_vector
            scale = np.linalg.norm(row)
            if scale > 0:
                row /= scale
            else:
                row = np.zeros(self.window)

            ndx = self.model.predict(row)[0]
            current = self.model.cluster_centers_[ndx, :]

            reconstructed_signal[i-self.window/2:i+self.window/2] += current * scale

        return reconstructed_signal[:, np.newaxis]


def main(args):
    # Read data.
    # Ref: https://github.com/mrahtz/sanger-machine-learning-workshop
    path = args[1]
    with open(path, 'rb') as input_file:
        data_raw = input_file.read()
    bytes = len(data_raw)
    shorts = bytes / 2
    unpack_string = '<%dh' % shorts
    x = np.array(struct.unpack(unpack_string, data_raw)).astype(float)

    # Anomaly detection
    x *= .02
    ekg = EKGAnomalyDetection(samples=50000)
    ekg.build_model(x)
    anomalies = ekg.detect_anomalies(x[:10000], .01)

    # Plotting
    xprime = ekg.reconstruct_signal(x[:10000])[:, 0]

    # Signal
    fig = plt.figure(figsize=(10, 5))
    plt.plot(x[:1000])
    plt.plot(xprime[:1000])
    plt.yticks([])
    plt.xticks([])
    plt.legend(['Observed', 'Predicted'])
    fig.savefig('signal.png')

    # Error
    fig = plt.figure(figsize=(10, 5))
    plt.plot(x[:1000] - xprime[:1000], c=sns.color_palette()[2])
    plt.xticks([])
    fig.savefig('error.png')

    # Anomaly
    max_error_i = max(anomalies, key=lambda x: np.abs(x.error)).index
    fig = plt.figure(figsize=(10, 5))
    plt.plot(x[max_error_i-10:max_error_i+10])
    plt.plot(xprime[max_error_i-10:max_error_i+10])
    plt.yticks([])
    plt.xticks([])
    fig.savefig('anomaly.png')

if __name__ == '__main__':
    main(sys.argv)
