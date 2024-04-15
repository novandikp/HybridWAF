import pandas as pd
import time
from feature_extraction.request_standarization import RequestStandarization


def transform_data(data) -> pd.DataFrame:
    request_standarization_list = []
    time_need = []
    if isinstance(data, pd.DataFrame):
        for index, row in data.iterrows():
            start = time.time()
            request = RequestStandarization(row.to_dict()).getFeatures()
            end = time.time()
            time_need.append(end - start)
            request_standarization_list.append(request)
    elif isinstance(data, list):
        for row in data:
            start = time.time()
            request = RequestStandarization(row).getFeatures()
            end = time.time()
            time_need.append(end - start)
            request_standarization_list.append(request)
    elif isinstance(data, dict):
        start = time.time()
        request = RequestStandarization(data).getFeatures()
        end = time.time()
        time_need.append(end - start)
        request_standarization_list.append(request)
    print("Time needed for request standarization: ", sum(time_need) / len(time_need))
    return pd.DataFrame(request_standarization_list)


def transform_data_with_time(data) -> pd.DataFrame:
    request_standarization_list = []
    time_need = []
    if isinstance(data, pd.DataFrame):
        for index, row in data.iterrows():
            start = time.time()
            request = RequestStandarization(row.to_dict()).getFeatures()
            end = time.time()
            time_need.append(end - start)
            request.append(end - start)
            request_standarization_list.append(request)
    elif isinstance(data, list):
        for row in data:
            start = time.time()
            request = RequestStandarization(row).getFeatures()
            end = time.time()
            time_need.append(end - start)
            request.append(end - start)
            request_standarization_list.append(request)
    elif isinstance(data, dict):
        start = time.time()
        request = RequestStandarization(data).getFeatures()
        end = time.time()
        time_need.append(end - start)
        request.append(end - start)
        request_standarization_list.append(request)
    print("Time needed for request standarization: ", sum(time_need) / len(time_need))
    data = pd.DataFrame(request_standarization_list)
    data.columns = list(data.columns[:-1]) + ["time"]
    return data
