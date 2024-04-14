import pandas as pd

from data_mining.ECML_Parser import saveDataECML
from data_mining.CSIC_Parser import saveDataCSIC


def saveTransformedData():
    saveDataECML()
    saveDataCSIC()