from posixpath import split
from numpy import average, tri
import requests
import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import time



drugDict = {}
edgeDict = {}
diseaseDict = {}

dimensions = 2


def avg(lst):
    sumNum = 0
    for x in range(0, len(lst)):
        sumNum+=float(lst[x])
    return float(sumNum) / float(len(lst))

def reloadData(samples):

    API_ENDPOINT = "http://robokopkg.renci.org/db/neo4j/tx"

    # Query names of links between 2 entities
    # Set body of request with neo4j query
    data = {
        "statements": [
            {
                "statement": "MATCH r=(a:`biolink:SmallMolecule`)-[b]->(c:`biolink:Disease`) where b:`biolink:treats` or b:`biolink:causes` RETURN distinct r, rand() as x order by x limit " + str(samples)
            }
        ]
    }

    response = requests.post(url = API_ENDPOINT, data = json.dumps(data), headers={'Content-Type': 'application/json', 'Accept': 'application/json;charset=UTF-8'})
    retData = json.loads(response.text)["results"][0]["data"]

    data = open("data.txt", "w")


    count = 0
    lineCount = 0

    triples = []

    for x in range(0, samples):
        drug = retData[x]["row"][0][0]["id"].split(":")[1].replace("CHEML", "")
        disease = retData[x]["row"][0][2]["id"].split(":")[1]
        predicate = retData[x]["row"][0][1]["predicate"].split(":")[1]

        drugNum = count
        edgeNum = count
        diseaseNum = count

        if not drug in drugDict:
            drugDict[drug] = [count, 1]
            drugNum = count
            count+=1
        else:
            temp = drugDict[drug]
            drugNum = temp[0]
            drugDict[drug] = [drugNum, temp[1]+1]
        
        if not disease in diseaseDict:
            diseaseDict[disease] = [count, 1]
            diseaseNum = count
            count+=1
        else:
            temp = diseaseDict[disease]
            diseaseNum = temp[0]
            diseaseDict[disease] = [diseaseNum, temp[1]+1]

        if not predicate in edgeDict:
            edgeDict[predicate] = [count]
            edgeNum = count
            count+=1
        else:
            temp = edgeDict[predicate]
            temp.append(count)
            edgeDict[predicate] = temp
            edgeNum = count
            count+=1

        tempTriple = [drugNum, edgeNum, diseaseNum, lineCount, drug, predicate, disease, [*edgeDict].index(predicate)]
        lineCount+=2
        triples.append(tempTriple)

    newTriples = []
    for x in range(0, len(triples)):
        if drugDict[triples[x][4]][1] >10:
            newTriples.append(triples[x])

    for x in range(0, len(newTriples)):
        data.write(str(newTriples[x][0]))
        data.write(" ")
        data.write(str(newTriples[x][1]))
        data.write(" 0 ")
        data.write(str(newTriples[x][3]))
        data.write("\n")
        lineCount+=1

        data.write(str(newTriples[x][1]))
        data.write(" ")
        data.write(str(newTriples[x][2]))
        data.write(" 1 ")
        data.write(str(newTriples[x][3]+1))
        data.write("\n")
    data.close()

reloadData(10000)
stream = os.popen('python3.8 transition.py')
time.sleep(10)

stream = os.popen('python3.8 edge2vec.py')
time.sleep(10)

file1 = open('vector.txt', 'r')
lines = file1.readlines()
 
count = 0
first = True

treatPoints = []
causePoints = []
for x in range(0, dimensions+1):
    treatPoints.append([])
    causePoints.append([])
label = []

xPoint = []
yPoint = []

# Strips the newline character
for line in lines:
    count += 1

    if first:
        first = False
    else:
        lineSplit = line.split(" ")
                
        good = False
        edgeType = ""
        for x in range(0, len([*edgeDict])):
            
            if int(lineSplit[0]) in edgeDict[[*edgeDict][x]]:
                good = True
                edgeType = [*edgeDict][x]
                break
        if good:
            label.append(edgeType)
            xPoint.append(float(lineSplit[1].replace("\n", "")))
            yPoint.append(float(lineSplit[2].replace("\n", "")))
            if edgeType == "causes":
                for x in range(1, len(lineSplit)):
                    causePoints[x].append(lineSplit[x].replace("\n", ""))
            else:
                for x in range(1, len(lineSplit)):
                    treatPoints[x].append(lineSplit[x].replace("\n", ""))

averageTreat = []
averageCause = []
for x in range(1, len(treatPoints)):
    averageTreat.append(avg(treatPoints[x]))
    averageCause.append(avg(causePoints[x]))

print(averageTreat)
print(averageCause)

from scipy.spatial import distance
print(distance.euclidean(averageTreat, averageCause))

df = pd.DataFrame(dict(xPoint=xPoint, yPoint=yPoint, label = label))

fig, ax = plt.subplots()

colors = {"treats":"red", "causes":"blue"}

ax.scatter(df['xPoint'], df['yPoint'], c=df['label'].map(colors))

plt.show()
