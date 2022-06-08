from posixpath import split
import requests
import json
import random

def reloadData(samples):

    source_edge = {}
    end_edge = {}
    edge_types = {}
    source_count = -1
    end_count = -1
    edge_count = -1

    API_ENDPOINT = "http://robokopkg.renci.org/db/neo4j/tx"

    # Query names of links between 2 entities
    # Set body of request with neo4j query
    data = {
        "statements": [
            {
                "statement": "MATCH r=(a:`biolink:SmallMolecule`)-[b]->(c:`biolink:Disease`) where b:`biolink:treats` or b:`biolink:causes_adverse_event` RETURN distinct r, rand() as x order by x limit " + str(samples)
            }
        ]
    }

    response = requests.post(url = API_ENDPOINT, data = json.dumps(data), headers={'Content-Type': 'application/json', 'Accept': 'application/json;charset=UTF-8'})
    retData = json.loads(response.text)["results"][0]["data"]

    data = open("data.txt", "w")

    total_sources = 0
    tempsources = []

    total_ends = 0
    tempends = []

    for x in range(0, samples):
        drug = retData[x]["row"][0][0]["id"].split(":")[1].replace("CHEML", "")
        disease = retData[x]["row"][0][2]["id"].split(":")[1]
        predicate = retData[x]["row"][0][1]["predicate"].split(":")[1]

        source =  drug
        end = disease

        if not source in tempsources:
            total_sources +=1
            tempsources.append(source)

        if not end in tempends:
            total_ends +=1
            tempends.append(end)

    end_count = total_sources-1

    # drug, predicate, disease
    for y in range(0, samples):

        drug = retData[y]["row"][0][0]["id"].split(":")[1].replace("CHEML", "")
        disease = retData[y]["row"][0][2]["id"].split(":")[1]
        predicate = retData[y]["row"][0][1]["predicate"].split(":")[1]

        source =  drug
        end = disease

        print(predicate)

        tempsource = -1
        tempend = -1
        tempedge = -1

        for x in range(0, len(source_edge.values())):
            if source_edge[x] == source:
                tempsource = x

        for x in range(0, len(edge_types.values())):
            if edge_types[x] == predicate:
                tempedge = x
        
        for x in range(total_sources, total_sources + len(end_edge.values())):
            if end_edge[x] == end:
                tempend = x

        if tempsource == -1:
            source_count+=1
            tempsource = source_count
            source_edge[tempsource] = source

        if tempend == -1:
            end_count+=1
            tempend = end_count
            end_edge[tempend] = end

        if tempedge == -1:
            edge_count+=1
            tempedge = edge_count
            edge_types[tempedge] = predicate

        data.write(str(tempsource))
        data.write(" ")
        data.write(str(tempend))
        data.write(" ")
        data.write(str(tempedge+3))
        data.write(" ")
        data.write(str(y))
        data.write("\n")

    data.close()

reloadData(600)