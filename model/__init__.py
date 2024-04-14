from cloudpickle import load
import os

MODEL_FOLDER_PATH = os.path.join(os.getcwd(), "model")
MODEL_PATH = os.path.join(MODEL_FOLDER_PATH, "best_psosvm.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_FOLDER_PATH, "label_encoder.pkl")
MINMAX_SCALER_PATH = os.path.join(MODEL_FOLDER_PATH, "minmax_scaler.pkl")

model = load(open(MODEL_PATH, "rb"))
labelEncoder = load(open(LABEL_ENCODER_PATH, "rb"))
minMax = load(open(MINMAX_SCALER_PATH, "rb"))



