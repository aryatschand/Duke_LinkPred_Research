import requests
import json
from terminaltables import AsciiTable
import time
start = time.time()

# Define 2 entities whose relationships will be measured
aEntity = "SmallMolecule"
bEntity = "Gene"
cEntity = "Disease"
targetLink = "treats"

# Define neo4j database endpoint
API_ENDPOINT = "http://robokopkg.renci.org/db/neo4j/tx"

# Query names of links between a-b entities and b-c entities
# Set body of request with neo4j query
data = {
    "statements": [
        {
            "statement": "MATCH p=(a:`biolink:" + aEntity + "`)-[r]->(b:`biolink:" + bEntity + "`) return distinct type(r)"
        },
        {
            "statement": "MATCH p=(a:`biolink:" + bEntity + "`)-[r]->(b:`biolink:" + cEntity + "`) return distinct type(r)"
        }
    ]
}

# Send post request and json parse response object
response = requests.post(url = API_ENDPOINT, data = json.dumps(data), headers={'Content-Type': 'application/json', 'Accept': 'application/json;charset=UTF-8'})
abJumpData = json.loads(response.text)["results"][0]["data"]
bcJumpData = json.loads(response.text)["results"][1]["data"]

# Create label array of all link names
abLabelArr = []
for x in range(0, len(abJumpData)):
    abLabelArr.append(abJumpData[x]["row"][0])
print(abLabelArr)
print("Total number of link types = " + str(len(abLabelArr)))
print()

# Create label array of all link names
bcLabelArr = []
for x in range(0, len(bcJumpData)):
    bcLabelArr.append(bcJumpData[x]["row"][0])
print(bcLabelArr)
print("Total number of link types = " + str(len(bcLabelArr)))
print()

# Query total number of links between 2 entities
data = {
            "statements": [
                {
                    "statement": "MATCH s=(a:`biolink:" + aEntity + "`)-[]->(b:`biolink:" + bEntity + "`)-[]->(c:`biolink:" + cEntity + "`) RETURN count(distinct [a,c])"
                },
                {
                    "statement": "MATCH s=(a:`biolink:" + aEntity + "`)-[:`biolink:" + targetLink + "`]->(c:`biolink:" + cEntity + "`) RETURN count(distinct [a,c])"
                }
            ]
        }
response = requests.post(url = API_ENDPOINT, data = json.dumps(data), headers={'Content-Type': 'application/json', 'Accept': 'application/json;charset=UTF-8'})
totalLinks = int(json.loads(response.text)["results"][0]["data"][0]["row"][0])
totalTargets = int(json.loads(response.text)["results"][1]["data"][0]["row"][0])
print("Total Links = " + str(totalLinks))
print()

printArr = []

# Loop through each combination of links to create an entry in print array
for x in range(0, len(abLabelArr)):
    for y in range(0, len(bcLabelArr)):
#for x in range(0, 3):
    #for y in range(0, 3):
            aLink = abLabelArr[x]
            bLink = bcLabelArr[y]
    
            # Query overlap between a-b and a-c Links
            data = {
                "statements": [
                    {
                        "statement": "MATCH s=(a:`biolink:" + aEntity + "`)-[:`" + aLink + "`]->(b:`biolink:" + bEntity + "`)-[:`" + bLink + "`]->(c:`biolink:" + cEntity + "`) return count(distinct [a,c])"
                    },
                    {
                        "statement": "MATCH (a:`biolink:" + aEntity + "`)-[:`" + aLink + "`]->(b:`biolink:" + bEntity + "`)-[:`" + bLink + "`]->(c:`biolink:" + cEntity + "`), s=(a)-[:`biolink:" + targetLink + "`]->(c) RETURN count(distinct [a,c])"
                    },
                ]
            }
            response = requests.post(url = API_ENDPOINT, data = json.dumps(data), headers={'Content-Type': 'application/json', 'Accept': 'application/json;charset=UTF-8'})
            
            # Positive = # of a-b links
            # True Positive = # of overlap of a-b and a-c links
            positive = int(json.loads(response.text)["results"][0]["data"][0]["row"][0])
            truePos = int(json.loads(response.text)["results"][1]["data"][0]["row"][0])
            print("(" + str(x) + "," + str(y) + ") True Positives with " + aLink + " and " + bLink + " = " + str(truePos))
            print()

            # False Positive = Positive - True Positive = (# of a-b links) - (# of overlap of a-b and a-c links)
            falsePos = positive - truePos

            # Negative = (# of links between 2 entities) - (# of a-b links)
            # False Negative = (# of a-c links) - True Positive = (# of a-c links) - (# of overlap of a-b and a-c links)
            falseNeg = totalTargets-truePos

            # True Negative = Negative - False Negative = (# of links between 2 entities) - (# of a-b links) - (# of a-c links) + (# of overlap of a-b and a-c links)
            trueNeg = totalLinks - positive - totalTargets - falseNeg

            # Sensitivity = True Positive/(True Positive + False Negative)
            sensitivity = round(truePos/(truePos + falseNeg), 3)

            # Specificity = True Negative/(True Negative + False Positive)
            specificity = round(trueNeg/(trueNeg + falsePos),3)

            # Precision = True Positive/(True Positive + False Positive)
            precision = round(truePos/(truePos + falsePos),3)

            printArr.append([aLink, bLink, truePos, falsePos, trueNeg, falseNeg, sensitivity, specificity, precision])

# Sort data by precision descending
printArr = sorted(printArr, key=lambda x: x[8], reverse=True)
printArr.insert(0, ["a-b", "a-c", "True Positive", "False Positive", "True Negative", "False Negative", "Sensitivity", "Specificity", "Precision"])
        
table = AsciiTable(printArr)
print(table.table)

# Calculate and output program runtime
end = time.time()
print(f"Runtime of the program is {end - start}")