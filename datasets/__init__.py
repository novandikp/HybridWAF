import os
import pandas as pd
import data_mining


def getFormattedECMLDatasets():
    dataset_location = os.path.join(os.getcwd(), "datasets", "formatted")
    if not os.path.exists(dataset_location):
        data_mining.saveDataECML()
    return pd.read_json(os.path.join(dataset_location, "ecml.json"))


def getFormattedCSICDatasets():
    dataset_location = os.path.join(os.getcwd(), "datasets", "formatted")
    if not os.path.exists(dataset_location):
        data_mining.saveDataCSIC()
    return pd.read_json(os.path.join(dataset_location, "csic.json"))


def getFormattedDatasets():
    csic_df = getFormattedCSICDatasets()
    ecml_df = getFormattedECMLDatasets()
    return pd.concat([csic_df, ecml_df], ignore_index=True)
