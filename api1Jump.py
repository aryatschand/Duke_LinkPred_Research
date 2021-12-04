import requests
import json
from terminaltables import AsciiTable
import time
start = time.time()

# Define 2 entities whose relationships will be measured
aEntity = "SmallMolecule"
bEntity = "Disease"
  
# Define neo4j database endpoint
API_ENDPOINT = "http://robokopkg.renci.org/db/neo4j/tx"

# Query names of links between 2 entities
# Set body of request with neo4j query
data = {
    "statements": [
        {
            "statement": "MATCH p=(a:`biolink:" + aEntity + "`)-[r]->(b:`biolink:" + bEntity + "`) return distinct type(r)"
        }
    ]
}
  
# Send post request and json parse response object
response = requests.post(url = API_ENDPOINT, data = json.dumps(data), headers={'Content-Type': 'application/json', 'Accept': 'application/json;charset=UTF-8'})
retData = json.loads(response.text)["results"][0]["data"]

# Create label array of all link names
labelArr = []
for x in range(0, len(retData)):
    labelArr.append(retData[x]["row"][0])
print(labelArr)
print("Total number of link types = " + str(len(labelArr)))

# Query total number of links between 2 entities
data = {
            "statements": [
                {
                    "statement": "MATCH s=(a:`biolink:" + aEntity + "`)-[]->(c:`biolink:" + bEntity + "`) RETURN distinct count([a,c])"
                }
            ]
        }
response = requests.post(url = API_ENDPOINT, data = json.dumps(data), headers={'Content-Type': 'application/json', 'Accept': 'application/json;charset=UTF-8'})
totalLinks = int(json.loads(response.text)["results"][0]["data"][0]["row"][0])
print("Total Links = " + str(totalLinks))

printArr = []

# Loop through each combination of links to create an entry in print array
for x in range(0, len(labelArr)):
    aLink = labelArr[x]

    # Query total number of a-b Links
    data = {
            "statements": [
                {
                    "statement": "MATCH s=(a:`biolink:" + aEntity + "`)-[:`" + aLink + "`]->(c:`biolink:" + bEntity + "`) RETURN distinct count([a,c])"
                }
            ]
        }
    response = requests.post(url = API_ENDPOINT, data = json.dumps(data), headers={'Content-Type': 'application/json', 'Accept': 'application/json;charset=UTF-8'})
    totalALinks = int(json.loads(response.text)["results"][0]["data"][0]["row"][0])

    for y in range(0, len(labelArr)):
        if (x != y):
            bLink = labelArr[y]
    
            # Query overlap between a-b and a-c Links
            data = {
                "statements": [
                    {
                        "statement": "MATCH (a:`biolink:" + aEntity + "`)-[:`" + aLink + "`]->(c:`biolink:" + bEntity + "`), s=(a)-[:`" + bLink + "`]->(c) RETURN distinct count([a,c])"
                    }
                ]
            }
            response = requests.post(url = API_ENDPOINT, data = json.dumps(data), headers={'Content-Type': 'application/json', 'Accept': 'application/json;charset=UTF-8'})
            
            # Positive = # of a-b links
            # True Positive = # of overlap of a-b and a-c links
            truePos = int(json.loads(response.text)["results"][0]["data"][0]["row"][0])
            print("(" + str(x) + "," + str(y) + ") True Positives between " + aLink + " and " + bLink + " = " + str(truePos))

            # False Positive = Positive - True Positive = (# of a-b links) - (# of overlap of a-b and a-c links)
            falsePos = totalALinks - truePos

            # Query total number of a-c links
            data = {
                "statements": [
                    {
                        "statement": "MATCH s=(a:`biolink:" + aEntity + "`)-[:`" + bLink + "`]->(c:`biolink:" + bEntity + "`) RETURN distinct count([a,c])"
                    }
                ]
            }
            response = requests.post(url = API_ENDPOINT, data = json.dumps(data), headers={'Content-Type': 'application/json', 'Accept': 'application/json;charset=UTF-8'})
            totalBLinks = int(json.loads(response.text)["results"][0]["data"][0]["row"][0])

            # Negative = (# of links between 2 entities) - (# of a-b links)
            # False Negative = (# of a-c links) - True Positive = (# of a-c links) - (# of overlap of a-b and a-c links)
            falseNeg = totalBLinks-truePos

            # True Negative = Negative - False Negative = (# of links between 2 entities) - (# of a-b links) - (# of a-c links) + (# of overlap of a-b and a-c links)
            trueNeg = totalLinks - totalALinks - falseNeg

            # Sensitivity = True Positive/(True Positive + False Negative)
            sensitivity = truePos/(truePos + falseNeg)

            # Specificity = True Negative/(True Negative + False Positive)
            specificity = trueNeg/(trueNeg + falsePos)

            # Precision = True Positive/(True Positive + False Positive)
            precision = truePos/(truePos + falseNeg)

            printArr.append([aLink, bLink, truePos, falsePos, trueNeg, falseNeg, sensitivity, specificity, precision])

# Sort data by precision descending
printArr = sorted(printArr, key=lambda x: x[8], reverse=True)
printArr.insert(0, ["a-b", "a-c", "True Positive", "False Positive", "True Negative", "False Negative", "Sensitivity", "Specificity", "Precision"])
        
table = AsciiTable(printArr)
print(table.table)

# Calculate and output program runtime
end = time.time()
print(f"Runtime of the program is {end - start}")