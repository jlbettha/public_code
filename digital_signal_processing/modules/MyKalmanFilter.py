import numpy as np
import mouse
import keyboard

# TODO: account for prior vs posterior


class MyKalmanFilter:
    def __init__(self, dt):
        self.A = np.array(
            [
                [1, 0, dt, 0, 0.5 * dt**2, 0],
                [0, 1, 0, dt, 0, 0.5 * dt**2],
                [0, 0, 1, 0, dt, 0],
                [0, 0, 0, 1, 0, dt],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )
        self.B = np.eye(6)
        self.H = np.eye(6)
        self.Q = 0.001 * np.eye(6)
        self.R = 0.1 * np.eye(6)
        self.c = np.zeros((6, 1))  # goal
        self.x = np.zeros((6, 1))  # initial state
        self.z = np.zeros((6, 1))  # noisy measurement | vel | accel
        self.P = np.zeros((6, 6))
        self.dt = dt

    def update(self):
        S = self.H @ P @ self.H.T + self.R
        K = P @ self.H.T @ np.linalg.pinv(S)
        y = z - self.H @ self.x
        self.x = self.x + K @ y
        P = (np.eye(6) - K @ self.H) @ P

    def predict(self):
        self.x = self.A @ self.x + self.B @ self.c
        P = self.A @ P @ self.A.T + self.Q
