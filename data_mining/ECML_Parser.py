import xml.etree.ElementTree as ET
import pandas as pd
import os
import json

data = []
dataset_location = os.path.join(os.getcwd(), "datasets", "ecml")
outuput_location = os.path.join(os.getcwd(), "datasets", "formatted")
root = ET.parse(os.path.join(dataset_location, "learning_dataset.xml")).getroot()

for child in root:
    request = {"id": child.attrib["id"]}
    for subchild in child:
        for subsub in subchild:
            request[subsub.tag] = subsub.text
    data.append(request)

df = pd.DataFrame(data)
df.drop(
    [
        "{http://www.example.org/ECMLPKDD}os",
        "{http://www.example.org/ECMLPKDD}webserver",
        "{http://www.example.org/ECMLPKDD}runningLdap",
        "{http://www.example.org/ECMLPKDD}runningSqlDb",
        "{http://www.example.org/ECMLPKDD}runningXpath",
    ],
    axis=1,
)


def getAttributeFromHeaders(headers, attribute):
    for h in headers:
        if h.startswith(attribute):
            return h[len(attribute) + 2 :].strip()
    return None


def getAllHeaders(headers):
    header = {}
    for h in headers:
        idx = h.find(":")
        header[h[:idx]] = h[idx + 2 :].strip()
    return header



def saveDataECML():
    formatedData = []
    for index, row in df.iterrows():
        request = {}
        if row["{http://www.example.org/ECMLPKDD}type"] == "Valid":
            request["type"] = "Valid"
        else:
            request["type"] = "Anomaly"
        headers = row["{http://www.example.org/ECMLPKDD}headers"].split("\n")
        request["host"] = getAttributeFromHeaders(headers, "Host")
        request["headers"] = getAllHeaders(headers)
        request["path"] = row["{http://www.example.org/ECMLPKDD}uri"]
        request["query"] = row["{http://www.example.org/ECMLPKDD}query"]
        request["method"] = row["{http://www.example.org/ECMLPKDD}method"]
        if request["method"] == "GET":
            request["body"] = {}
        else:
            request["body"] = row["{http://www.example.org/ECMLPKDD}body"]
        formatedData.append(request)

    # calculate Valid and Anomaly
    print("ECML Dataset: ")
    print("Valid: ", len([x for x in formatedData if x["type"] == "Valid"]))
    print("Anomaly: ", len([x for x in formatedData if x["type"] == "Anomaly"]))
    # export to json
    with open(os.path.join(outuput_location, "ecml.json"), "w") as f:
        print("Save data to ", os.path.join(outuput_location, "ecml.json"))
        json.dump(formatedData, f)
