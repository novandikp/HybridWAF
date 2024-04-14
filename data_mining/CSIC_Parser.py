import urllib.parse
import json
import os

dataset_location = os.path.join(os.getcwd(), "datasets", "csic")
outuput_location = os.path.join(os.getcwd(), "datasets", "formatted")

normal_file_parse = os.path.join(dataset_location, "normalTrafficTest.txt")
anomaly_file_parse = os.path.join(dataset_location, "anomalousTrafficTest.txt")


def parse_file(file_in, type):
    fin = open(file_in)
    lines = fin.readlines()
    res = []
    requests = {}
    for i in range(len(lines)):
        line = lines[i].strip()
        if line.startswith("GET") or line.startswith("POST") or line.startswith("PUT") or line.startswith("DELETE"):
            if i > 0:
                requests['type'] = type
                res.append(requests)
                requests = {}

            requests['method'] = line.split(' ')[0]
            requests['path'] = line.split(' ')[1]
            requests["headers"] = {}
            requests["headers"]["Host"] = "http://localhost:8080"
            if line.startswith("GET") and '?' in line:
                requests['query'] = line.split(' ')[1].split('?')[1]
                requests['query'] = dict(urllib.parse.parse_qsl(requests['query']))
                requests['path'] = line.split(' ')[1].split('?')[0]
                requests['path'] = requests['path'].replace('http://localhost:8080', '')
            else:
                requests['path'] = line.split(' ')[1].split('?')[0]
                requests['path'] = requests['path'].replace('http://localhost:8080', '')
        # check line is header
        if ': ' in line:
            header = line.split(':')
            requests["headers"][header[0]] = header[1].strip()

        # get body
        j = 1
        if requests['method'] == 'POST' or requests['method'] == 'PUT':
            while i + j < len(lines):
                if lines[i + j].startswith("Content-Length"):
                    break
                j += 1
            j += 1
            if i + j < len(lines):
                requests['body'] = lines[i + j + 1].strip()
                requests['body'] = dict(urllib.parse.parse_qsl(requests['body']))

        if i == len(lines) - 1:
            res.append(requests)
    fin.close()
    return res


def saveDataCSIC():
    valid_data = parse_file(normal_file_parse, "Valid")
    anomaly_data = parse_file(anomaly_file_parse, "Anomaly")
    print("CSIC Dataset: ")
    print("Valid: ", len(valid_data))
    print("Anomaly: ", len(anomaly_data))
    # merge data
    data = valid_data + anomaly_data
    # save data
    with open(os.path.join(outuput_location, "csic.json"), "w") as outfile:
        json.dump(data, outfile)
        print("Save data to ", os.path.join(outuput_location, "csic.json"))
