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

dimensions = 100


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

reloadData(20000)
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
for x in range(0, 100):
    columns.append(str(x))
columns.append("edge")

print(points)
print(columns)
import numpy as np
dataset = pd.DataFrame(np.array(points), columns=columns)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 100].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt

knn = KNeighborsClassifier(n_neighbors = 10)
knn.fit(X_train,y_train)

y_scores = knn.predict_proba(X_test)
fpr, tpr, threshold = roc_curve(y_test, y_scores[:, 1])
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve of kNN')
plt.show()

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
'''

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score


classifier = OneVsRestClassifier(
    svm.SVC()
)
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(0):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

print(fpr)
print(tpr)
print(y_test)
print(y_score)

plt.figure()
lw = 2
plt.plot(
    fpr['micro'],
    tpr['micro'],
    color="darkorange",
    lw=lw,
    label="ROC curve (area = %0.2f)" % roc_auc[2],
)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic example")
plt.legend(loc="lower right")
plt.show()

'''


from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns

xPoint = []
yPoint = []

maxVals = []
for x in range(0, len(points)):
    for y in range(0, len(points[x])):
        maxVals.append(abs(max(points[x], key=abs)))
    #print(points[x])
    xPoint.append(points[x][10])
    yPoint.append(points[x][21])

'''
df = pd.DataFrame(dict(xPoint=maxVals, label = label))


#df = pd.DataFrame(dict(xPoint=xPoint, yPoint=yPoint, label = label))
fig, ax = plt.subplots()

colors = {"treats":"red", "causes":"blue"}

ax.scatter(df['xPoint'], c=df['label'].map(colors))

#ax.scatter(df['xPoint'], df['yPoint'], c=df['label'].map(colors))

plt.show()
'''


df = pd.DataFrame(dict(xPoint=xPoint, yPoint=yPoint, label = label))
fig, ax = plt.subplots()

colors = {"treats":"red", "causes":"blue"}

ax.scatter(df['xPoint'], df['yPoint'], c=df['label'].map(colors))

'''

# We want to get TSNE embedding with 2 dimensions
n_components = 2
tsne = TSNE(n_components)
tsne_result = tsne.fit_transform(points)

xPoint = []
yPoint = []


for x in range(0, len(tsne_result)):
    xPoint.append(tsne_result[x][0])
    yPoint.append(tsne_result[x][1])

df = pd.DataFrame(dict(xPoint=xPoint, yPoint=yPoint, label = label))

fig, ax = plt.subplots()

colors = {"treats":"red", "causes":"blue"}

ax.scatter(df['xPoint'], df['yPoint'], c=df['label'].map(colors))

plt.show()
'''