# utils/kalman_filter.py
import numpy as np

class KalmanFilter:
    """
    Simple 2-D bounding-box Kalman filter:
    state = [cx, cy, w, h, dcx, dcy, dw, dh]
    returns 4-box: [x1, y1, x2, y2]
    """
    def __init__(self):
        dt = 1.0
        self.F = np.array([
            [1, 0, 0, 0, dt, 0,  0,  0],
            [0, 1, 0, 0, 0,  dt, 0,  0],
            [0, 0, 1, 0, 0,  0,  dt, 0],
            [0, 0, 0, 1, 0,  0,  0,  dt],
            [0, 0, 0, 0, 1,  0,  0,  0],
            [0, 0, 0, 0, 0,  1,  0,  0],
            [0, 0, 0, 0, 0,  0,  1,  0],
            [0, 0, 0, 0, 0,  0,  0,  1]
        ])
        self.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])
        self.Q = np.eye(8) * 1e-2
        self.R = np.eye(4) * 1e-1
        self.x = None
        self.P = None

    def _init(self, bbox):
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w  = bbox[2] - bbox[0]
        h  = bbox[3] - bbox[1]
        self.x = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32)
        self.P = np.eye(8) * 10.

    def predict(self):
        if self.x is None:
            return None
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        cx, cy, w, h = self.x[:4]
        x1, y1, x2, y2 = cx-w/2, cy-h/2, cx+w/2, cy+h/2
        return [x1, y1, x2, y2]

    def update(self, bbox):
        if self.x is None:
            self._init(bbox)
            return bbox
        z = np.array(bbox, dtype=np.float32)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P
        cx, cy, w, h = self.x[:4]
        x1, y1, x2, y2 = cx-w/2, cy-h/2, cx+w/2, cy+h/2
        return [x1, y1, x2, y2]