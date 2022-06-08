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
    '''
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
    jsonFile = open("jsonFile.txt", "w")
    jsonFile.write(response.text)
    jsonFile.close()
    '''
    jsonFile = open("jsonFile.txt", "r")
    contents = jsonFile.read()


    retData = json.loads(contents)["results"][0]["data"]

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
        if drugDict[triples[x][4]][1] >5:
            newTriples.append(triples[x])

    for x in range(0, len(newTriples)):
        data.write(str(newTriples[x][0]))
        data.write(" ")
        data.write(str(newTriples[x][1]))
        data.write(" ")
        data.write(str(newTriples[x][7]))
        data.write(" ")
        data.write(str(newTriples[x][3]))
        data.write("\n")
        lineCount+=1

        data.write(str(newTriples[x][1]))
        data.write(" ")
        data.write(str(newTriples[x][2]))
        data.write(" ")
        data.write(str(newTriples[x][7]+3))
        data.write(" ")
        data.write(str(newTriples[x][3]+1))
        data.write("\n")
    data.close()

reloadData(5000)
shell = "python3.8 transition.py --dimensions=" + str(dimensions)
stream = os.popen(shell)
time.sleep(5)

shell = "python3.8 edge2vec.py --dimensions=" + str(dimensions)
stream = os.popen(shell)
time.sleep(5)

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

points = []

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
            temp = []
            for x in range(1, len(lineSplit)):
                temp.append(float(lineSplit[x].replace("\n", "")))
            if edgeType == "causes":
                temp.append(1)
            else:
                temp.append(-1)
            points.append(temp)

columns = []
for x in range(0, dimensions):
    columns.append(str(x))
columns.append("edge")

import numpy as np
dataset = pd.DataFrame(np.array(points), columns=columns)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, dimensions].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

#X_train = scaler.transform(X_train)
#X_test = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

print('Accuracy of GNB classifier on training set: {:.2f}'
     .format(gnb.score(X_train, y_train)))
print('Accuracy of GNB classifier on test set: {:.2f}'
     .format(gnb.score(X_test, y_test)))

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(X_test, y_test)))

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier().fit(X_train, y_train)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))

from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, y_train)

print('Accuracy of SVM classifier on training set: {:.2f}'
     .format(svm.score(X_train, y_train)))
print('Accuracy of SVM classifier on test set: {:.2f}'
     .format(svm.score(X_test, y_test)))


import pandas as pd
import seaborn as sns

xPoint = []
yPoint = []

maxVals = []
maxCause = 0
for x in range(0, len(points)):
    tempPoints = points[x][0:-1]
    maxVals.append(abs(max(tempPoints, key=abs)))
        
    #print(points[x])
    #xPoint.append(points[x][10])
    yPoint.append(points[x][-1])

    if points[x][-1] == 1 and abs(max(tempPoints, key=abs)) > maxCause:
        maxCause = abs(max(tempPoints, key=abs))

# cause under threshold, total cause, treats under threshold, total treats
totalArray = [0, 0, 0, 0]
for x in range(0, len(maxVals)):
    if label[x] == "causes":
        totalArray[1]+=1
        if maxVals[x] <= maxCause:
            totalArray[0]+=1
    else:
        totalArray[3]+=1
        if maxVals[x] <= maxCause:
            totalArray[2]+=1

print(1-float(totalArray[2]/float(totalArray[3])))
    

#print(maxVals)
#print(len(label))
#print(totalArray)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

intEdges = []
for x in range(0, len(points)):
    intEdges.append(points[x][-1])

tempThreshold = max(maxVals)
steps = tempThreshold/100.0
fpr = []
tpr = []
for x in range(0, len(maxVals)):
    # cause over threshold, total cause, treats over threshold, total treats
    totalArray = [0, 0, 0, 0]
    tempThreshold = maxVals[len(maxVals)-1-x]
    for x in range(0, len(maxVals)):
        if label[x] == "causes":
            totalArray[1]+=1
            if maxVals[x] >= tempThreshold:
                totalArray[0]+=1
        else:
            totalArray[3]+=1
            if maxVals[x] >= tempThreshold:
                totalArray[2]+=1

    

    tpr.append(float(totalArray[0]+totalArray[2])/float(totalArray[1]+totalArray[3]))
    fpr.append(float(totalArray[0])/float(totalArray[0]+totalArray[2]))

    #tempThreshold-=steps
#plt.plot(tpr,fpr)
#plt.ylabel('True Positive Rate')
#plt.xlabel('False Positive Rate')
#plt.show()

from sklearn.model_selection import GridSearchCV  
from sklearn.linear_model import SGDClassifier 
from sklearn.metrics import roc_curve, auc




X_train,X_test,y_train,y_test = train_test_split(maxVals,intEdges,test_size=0.3,random_state=0) 
X_train= np.array(X_train).reshape(-1, 1)
X_test = np.array(X_test).reshape(-1, 1)

model = SGDClassifier(loss='hinge')
model.fit(X_train, y_train)
# roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class, not the predicted outputs.

y_train_pred = model.decision_function(X_train)    
y_test_pred = model.decision_function(X_test) 

train_fpr, train_tpr, tr_thresholds = roc_curve(y_train, y_train_pred)
test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_test_pred)

plt.grid()

plt.plot(train_fpr, train_tpr, label=" AUC TRAIN ="+str(auc(train_fpr, train_tpr)))
plt.plot(test_fpr, test_tpr, label=" AUC TEST ="+str(auc(test_fpr, test_tpr)))
plt.plot([0,1],[0,1],'g--')
plt.legend()
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("AUC(ROC curve)")
plt.grid(color='black', linestyle='-', linewidth=0.5)
plt.show()




X_train,X_test,y_train,y_test = train_test_split(maxVals,intEdges,test_size=0.3,random_state=0) 
X_train= np.array(X_train).reshape(-1, 1)
X_test = np.array(X_test).reshape(-1, 1)

#instantiate the model
log_regression = LogisticRegression()


#fit the model using the training data
log_regression.fit(X_train,y_train)

y_pred_proba = log_regression.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)

print(fpr)
print(tpr)
#create ROC curve
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


tempY = []
for x in range(0, len(label)):
    tempY.append(0)
df = pd.DataFrame(dict(xPoint=maxVals, yPoint=yPoint, label = label))


#df = pd.DataFrame(dict(xPoint=xPoint, yPoint=yPoint, label = label))
fig, ax = plt.subplots()

colors = {"treats":"red", "causes":"blue"}

ax.scatter(df['xPoint'], df['yPoint'], c=df['label'].map(colors))

#ax.scatter(df['xPoint'], df['yPoint'], c=df['label'].map(colors))

plt.show()


'''
df = pd.DataFrame(dict(xPoint=xPoint, yPoint=yPoint, label = label))
fig, ax = plt.subplots()

colors = {"treats":"red", "causes":"blue"}

ax.scatter(df['xPoint'], df['yPoint'], c=df['label'].map(colors))
plt.show()
'''