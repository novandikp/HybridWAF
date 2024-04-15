from cloudpickle import load
import os

MODEL_FOLDER_PATH = os.path.join(os.getcwd(), "model")
MODEL_PATH = os.path.join(MODEL_FOLDER_PATH, "best_psosvm.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_FOLDER_PATH, "label_encoder.pkl")
MINMAX_SCALER_PATH = os.path.join(MODEL_FOLDER_PATH, "minmax_scaler.pkl")

model = None
labelEncoder = None
minMax = None

if os.path.exists(MODEL_PATH):
    model = load(open(MODEL_PATH, "rb"))

if os.path.exists(LABEL_ENCODER_PATH):
    labelEncoder = load(open(LABEL_ENCODER_PATH, "rb"))

if os.path.exists(MINMAX_SCALER_PATH):
    minMax = load(open(MINMAX_SCALER_PATH, "rb"))



