import numpy as np


# ref: https://en.wikipedia.org/wiki/Kalman_filter
class DiscreteKF:
    def __init__(self, A_k, B_k, H_k, Q_k, R_k):
        self.Ak = A_k
        self.Bk = B_k
        self.Hk = H_k
        self.Qk = Q_k
        self.Rk = R_k
        self.xk_minus = None
        self.Pk_minus = None

    def init_filter(self, x0, P0):
        self.xk_minus = x0
        self.Pk_minus = P0

    def predict(self, uk_minus):
        if self.xk_minus is None:
            raise Exception(
                "Error: initalize the filter first by calling init_filter ..."
            )
        if self.Pk_minus is None:
            raise Exception(
                "Error: initalize the filter first by calling init_filter ..."
            )

        # Project state ahead.
        pred = self.Ak @ self.xk_minus + self.Bk @ uk_minus
        assert pred.shape == (2, 1)
        self.xk_minus = pred

        # Project error cov ahead.
        self.Pk_minus = self.Ak @ self.Pk_minus @ np.transpose(self.Ak) + self.Qk

    def correct(self, zk):
        if self.xk_minus is None:
            raise Exception(
                "Error: initalize the filter first by calling init_filter ..."
            )
        if self.Pk_minus is None:
            raise Exception(
                "Error: initalize the filter first by calling init_filter ..."
            )

        # Compute kalman gain.
        Kyy = self.Hk @ self.Pk_minus @ np.transpose(self.Hk) + self.Rk
        gain = self.Pk_minus @ np.transpose(self.Hk)
        gain = gain @ np.linalg.inv(Kyy)

        # Innovate (correct estimate based on new measurement).
        zk_minus = self.Hk @ self.xk_minus
        corr = self.xk_minus + gain @ (zk - zk_minus)
        assert corr.shape == (2, 1)
        self.xk_minus = corr

        # Correct cov err.
        I = np.eye(self.Pk_minus.shape[0])
        self.Pk_minus = (I - gain @ self.Hk) @ np.linalg.inv(self.Pk_minus)

    def get_state_estim(self):
        return self.xk_minus


# ref: https://en.wikipedia.org/wiki/Extended_Kalman_filter
class DiscreteEKF:
    def __init__(self, f, h, A_k, H_k, Q_k, R_k):
        self.f = f  # nonlinear dynamics model
        self.h = h  # nonlinear measurement model
        self.Ak = A_k  # jacobian df/dx(x, y)
        self.Hk = H_k  # jacobian dh/dx(x)
        self.Qk = Q_k
        self.Rk = R_k
        self.xk_minus = None
        self.Pk_minus = None

    def init_filter(self, x0, P0):
        self.xk_minus = x0
        self.Pk_minus = P0

    def predict(self, uk_minus):
        if self.xk_minus is None:
            raise Exception(
                "Error: initalize the filter first by calling init_filter ..."
            )
        if self.Pk_minus is None:
            raise Exception(
                "Error: initalize the filter first by calling init_filter ..."
            )

        # Project state ahead.
        self.xk_minus = self.f(self.xk_minus, uk_minus)
        assert self.xk_minus.shape == (2, 1)

        # Compute Jacobian of non-linear state transition model.
        Ak = self.Ak(self.xk_minus, uk_minus)
        assert Ak.shape == (2, 2)

        # Project error cov ahead.
        self.Pk_minus = Ak @ self.Pk_minus @ np.transpose(Ak) + self.Qk
        assert self.Pk_minus.shape == (2, 2)

    def correct(self, zk):
        if self.xk_minus is None:
            raise Exception(
                "Error: initalize the filter first by calling init_filter ..."
            )
        if self.Pk_minus is None:
            raise Exception(
                "Error: initalize the filter first by calling init_filter ..."
            )

        # Compute Jacobian of non-linear measurement transition model.
        Hk = self.Hk(self.xk_minus)
        assert Hk.shape == (2, 2)

        # Compute kalman gain.
        Kyy = Hk @ self.Pk_minus @ np.transpose(Hk) + self.Rk
        gain = self.Pk_minus @ np.transpose(Hk) @ np.linalg.inv(Kyy)

        # Innovate.
        zk_minus = self.h(self.xk_minus)
        assert zk_minus.shape == (2, 1)
        self.xk_minus = self.xk_minus + gain @ (zk - zk_minus)
        assert self.xk_minus.shape == (2, 1)

        # Correct cov err.
        I = np.eye(self.Pk_minus.shape[0])
        self.Pk_minus = (I - gain @ Hk) @ np.linalg.inv(self.Pk_minus)
        assert self.Pk_minus.shape == (2, 2)

    def get_state_estim(self):
        return self.xk_minus
