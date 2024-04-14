import numpy as np


class LabelEncoder:
    def __init__(self):
        pass

    def transform(self, data):
        if isinstance(data, str):
            return 1 if data == "Valid" else -1
        return np.where(data == "Valid", 1, -1)

    def inverse_transform(self, data):
        if isinstance(data, (int, float)):
            return "Valid" if data == 1 else "Anomaly"
        return np.where(data == 1, "Valid", "Anomaly")
