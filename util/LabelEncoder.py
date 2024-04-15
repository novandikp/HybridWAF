import numpy as np


class LabelEncoder:
    def __init__(self):
        self.neg = 0
        pass

    def transform(self, data):
        if isinstance(data, str):
            return 1 if data == "Valid" else self.neg
        return np.where(data == "Valid", 1, self.neg)

    def inverse_transform(self, data):
        if isinstance(data, (int, float)):
            return "Valid" if data == 1 else "Anomaly"
        return np.where(data == 1, "Valid", "Anomaly")
