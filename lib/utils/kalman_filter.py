import numpy as np


class KalmanFilter(object):
    def __init__(self, dim=2, dt=1., uncertainty_x=0.05, uncertainty_v=0.00625):
        # For x: [x1, x2, ..., vx1, vx2, ...], where vx is velocity of change in x
        # For z: [z1, z2, ...]
        self.dim = dim

        self.uncertainty_x = uncertainty_x
        self.uncertainty_v = uncertainty_v

        # State transition matrix
        self.F = np.eye(2 * dim, 2 * dim)
        # Set all x_new = x + vx * 1
        for i in range(dim):
            self.F[i][i + dim] = dt

        std = np.r_[np.ones(dim) * uncertainty_x, np.ones(dim) * uncertainty_v]

        # Process uncertainty
        self.Q = np.diag(np.square(std))

        # Measurement function, convert x to z
        self.H = np.eye(dim, 2 * dim)

        # Covariance matrix
        self.P = np.diag(np.square(std))

        # State uncertainty
        std_z = np.ones(dim) * uncertainty_x
        self.R = np.diag(np.square(std_z))

    def initiate(self, measurement):
        # Use first value as initial value
        v = np.zeros_like(measurement)
        self.x = np.r_[measurement, v].astype(float)

        std = np.r_[np.ones(self.dim) * self.uncertainty_x, np.ones(self.dim) * self.uncertainty_v]
        # Covariance matrix
        self.P = np.diag(np.square(std))

    def predict(self, warp=None):
        """
            warp: numpy array of shape (2, 3)
        """
        if warp is not None:
            x = np.dot(self.F, self.x)
            x1 = np.array([[x[0], x[1], 1]]).T
            x2 = np.array([[x[2], x[3], 1]]).T
            x1_n = np.dot(warp, x1).reshape((1, 2))
            x2_n = np.dot(warp, x2).reshape((1, 2))
            x[:self.dim] = np.concatenate((x1_n, x2_n), axis=1)
            self.x = x
        else:
            self.x = np.dot(self.F, self.x)
        self.P = np.dot(self.F, self.P).dot(self.F.T) + self.Q
        return self.x[:self.dim]

    def update(self, z):
        self.S = np.dot(self.H, self.P).dot(self.H.T) + self.R
        self.K = np.dot(self.P, self.H.T).dot(np.linalg.inv(self.S))
        y = z - np.dot(self.H, self.x)
        self.x += np.dot(self.K, y)
        self.P = self.P - np.dot(self.K, self.H).dot(self.P)
        return self.x[:self.dim], self.x[self.dim:]
