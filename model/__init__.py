from cloudpickle import load
import os


def load_model(varian: str) -> tuple:
    MODEL_FOLDER_PATH = os.path.join(os.getcwd(), "model")
    MODEL_PATH = os.path.join(MODEL_FOLDER_PATH, f"model_{varian}.pkl")
    MINMAX_SCALER_PATH = os.path.join(MODEL_FOLDER_PATH, f"minmax_scaler{varian}.pkl")
    model = load(open(MODEL_PATH, "rb"))
    minmax = load(open(MINMAX_SCALER_PATH, "rb"))
    return model, minmax
