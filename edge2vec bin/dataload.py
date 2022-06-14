from posixpath import split
from numpy import average, tri
import requests
import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import time

# Dictionaries mapping node and edge IDs in data.txt file to actual drug, predicate, or disease
drugDict = {}
predicateDict = {}
diseaseDict = {}

# Number of dimensions in embedding vector
dimensions = 100

# Load data from query or local file into data structures
def reloadData(samples):
    
    # Query data from Robokopkg Neo4j online database
    # Query is not necessary after response is stored locally to jsonFile.txt 
    '''

    # Set endpoint of response
    API_ENDPOINT = "http://robokopkg.renci.org/db/neo4j/tx"

    # Query treats and causes relationships between any SmallMolecule and Disease entities
    # Set body of request with Neo4j query
    data = {
        "statements": [
            {
                "statement": "MATCH r=(a:`biolink:SmallMolecule`)-[b]->(c:`biolink:Disease`) where b:`biolink:treats` or b:`biolink:causes` RETURN distinct r, rand() as x order by x limit " + str(samples)
            }
        ]
    }

    # Store response to jsonFile.txt so repeating query does not need to be made
    response = requests.post(url = API_ENDPOINT, data = json.dumps(data), headers={'Content-Type': 'application/json', 'Accept': 'application/json;charset=UTF-8'})
    jsonFile = open("jsonFile.txt", "w")
    jsonFile.write(response.text)
    jsonFile.close()
    '''

    # Read contents of jsonFile.txt, parse to data location, and store in data structure
    jsonFile = open("jsonFile.txt", "r")
    contents = jsonFile.read()
    retData = json.loads(contents)["results"][0]["data"]

    # Data.txt contains the formatted data for edge2vec embeddings input
    # Redefine triples with predicate relationship as a node
    # Drug to predicate has “a” edge and predicate to disease has “b” edge
    # File format - source ID, target ID, a/b edge, entry ID
    data = open("data.txt", "w")

    # Counters for incremential IDs within data.txt
    count = 0
    lineCount = 0

    # List with information about each triple in dataset
    triples = []

    # Iterate through each triple returned from Robokopkg query
    for x in range(0, samples):

        # Parse and clean data to extract predicate and unique bio indentifier for each drug and disease
        # This unique identifier is different than the IDs in data.txt file 
        drug = retData[x]["row"][0][0]["id"].split(":")[1].replace("CHEML", "")
        disease = retData[x]["row"][0][2]["id"].split(":")[1]
        predicate = retData[x]["row"][0][1]["predicate"].split(":")[1]

        # Initialize drug, predicate, and disease IDs for data.txt file (will be changed)
        drugNum = count
        predicateNum = count
        diseaseNum = count

        # For the drug, egde, and disease in each triple, check if it has been seen
        # If it has been seen, query and save ID
        # If not, increment ID and append it to cumulative dictionary mapping to unique identifier
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

        if not predicate in predicateDict:
            predicateDict[predicate] = [count]
            predicateNum = count
            count+=1
        else:
            temp = predicateDict[predicate]
            temp.append(count)
            predicateDict[predicate] = temp
            predicateNum = count
            count+=1

        # Store drug, predicate, and disease IDs and bio identifiers to triples list
        tempTriple = [drugNum, predicateNum, diseaseNum, lineCount, drug, predicate, disease, [*predicateDict].index(predicate)]
        lineCount+=2
        triples.append(tempTriple)

    # Create new triples array that removes potential outliers
    # Only keep triples where the drug has a causes or treats relationship with >5 diseases
    newTriples = []
    for x in range(0, len(triples)):
        if drugDict[triples[x][4]][1] >5:
            newTriples.append(triples[x])

    # Write IDs corresponding to redefined triple to data.txt file
    for x in range(0, len(newTriples)):
        # First half of triple representing drug to predicate
        data.write(str(newTriples[x][0]))
        data.write(" ")
        data.write(str(newTriples[x][1]))
        data.write(" ")
        data.write(str(newTriples[x][7]))
        data.write(" ")
        data.write(str(newTriples[x][3]))
        data.write("\n")
        lineCount+=1

        # Second half of triple representing predicate to disease
        data.write(str(newTriples[x][1]))
        data.write(" ")
        data.write(str(newTriples[x][2]))
        data.write(" ")
        data.write(str(newTriples[x][7]+1))
        data.write(" ")
        data.write(str(newTriples[x][3]+1))
        data.write("\n")
    data.close()

# Reload data with 5000 triple limit
# There will be 5000 triples in the data set, as a portion of the drugs do not have >5 predicate
reloadData(5000)

# Define and execute shell command to create transition matrix for edge embedding model
shell = "python3.8 transition.py --dimensions=" + str(dimensions)
stream = os.popen(shell)
time.sleep(5)

# Define and execute shell command to create embeddings with transition matrix and data.txt input data
shell = "python3.8 edge2vec.py --dimensions=" + str(dimensions)
stream = os.popen(shell)
time.sleep(5)

# Open outputted vector.txt file containing embeddings
file1 = open('vector.txt', 'r')

lines = file1.readlines()
count = 0
first = True

# Create list for predicate label and magnitude of embedding in each dimension
label = []
points = []

# Iterate through each embedding vector
# Beacuse of the redefined triple structure, 2 embeddings represent a single triple
for line in lines:
    count += 1

    if first:
        first = False
    else:
        # In a valid embedding vector, split line into n index list (n = # of dimension)
        lineSplit = line.split(" ")

        good = False
        predicateType = ""

        # Check if predicate ID is mapped to a valid predicate
        for x in range(0, len([*predicateDict])):
            if int(lineSplit[0]) in predicateDict[[*predicateDict][x]]:
                good = True

                # Index predicateDict as list to save type
                predicateType = [*predicateDict][x]
                break
        
        # If a valid predicate is found, append data to label and points lists
        if good:
            # Append predicate type to label list
            label.append(predicateType)

            temp = []

            # Append data to points - m x (n+1) 2D array
            # m = number of triples with drug satisfying >5 predicates
            # Each m is an edge embedding of the predicate of the triple
            # n = # of dimensions
            # n+1 index of each m index is an integer representation of each predicate
            for x in range(1, len(lineSplit)):
                temp.append(float(lineSplit[x].replace("\n", "")))
            if predicateType == "causes":
                temp.append(-1)
            else:
                temp.append(1)
            points.append(temp)

# Columns list needed for sklearn model dataset
columns = []
for x in range(0, dimensions):
    columns.append(str(x))
columns.append("edge")

# Prepare sklearn dataset
# Create pandas dataframe using numpy array of points and columns list as input
import numpy as np
dataset = pd.DataFrame(np.array(points), columns=columns)

# X data is embedding vectors
X = dataset.iloc[:, :-1].values

# Y data is predicate type
y = dataset.iloc[:, dimensions].values

# Split dataset into X and Y data, and use 70/30 train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# Train and test Standard Scalar model in classification of predicates with embedding data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

# Train and test KNN model in classification of predicates with embedding data
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Train and test GaussianNB model in classification of predicates with embedding data
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
print('Accuracy of GNB classifier on training set: {:.2f}'
    .format(gnb.score(X_train, y_train)))
print('Accuracy of GNB classifier on test set: {:.2f}'
    .format(gnb.score(X_test, y_test)))

# Train and test different variation of KNN model in classification of predicates with embedding data
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'
    .format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
    .format(knn.score(X_test, y_test)))

# Train and test Decision Tree Classifier model in classification of predicates with embedding data
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier().fit(X_train, y_train)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
    .format(clf.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
    .format(clf.score(X_test, y_test)))

# Train and test Support Vector Classifier model in classification of predicates with embedding data
from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, y_train)
print('Accuracy of SVM classifier on training set: {:.2f}'
    .format(svm.score(X_train, y_train)))
print('Accuracy of SVM classifier on test set: {:.2f}'
    .format(svm.score(X_test, y_test)))


import pandas as pd
import seaborn as sns

# List of integer representation of predicate types
yPoint = []

# maxVals contains the Max Absolute Value (MAV) for each triple
# MAV is the highest magnitude dimension in each triple embedding vector
maxVals = []

# Iterate through and calculate MAV for each edge embedding
for x in range(0, len(points)):
    tempPoints = points[x][0:-1]
    maxVals.append(abs(max(tempPoints, key=abs)))
    yPoint.append(points[x][-1])


# List of False and True Positive rates at each threshold iteration
falsepos = []
truepos = []

# Total number of positive (treats edges) and negatives (causes edges)
P = label.count("treats")
N = label.count("causes")

# Create new list of sorted maxVals
sortThresholds = maxVals.copy()
sortThresholds.sort()

# Iterate through sorted maxVals as thresholds for ROC curve
for threshold in sortThresholds:
    FP = 0
    TP = 0

    # Find number of True and False Positives above each threshold
    for x in range(len(maxVals)):
        if (maxVals[x] >= threshold):
            if (label[x] == "treats"):
                TP = TP+1
            else:
                FP = FP+1
    
    # False Positive rate = # of False Positives above threshold / Total Negatives
    falsepos.append(float(FP)/float(N))

    # True Positive rate = # of True Positives above threshold / Total Positives 
    truepos.append(float(TP)/float(P))

# Scale AUC
auc = -1 * np.trapz(truepos, falsepos)

# Plot ROC curve with False Positive rates on X-axis and True Positive rates on Y-axis
plt.plot(falsepos, truepos, linestyle='--', marker='o', color='darkorange', lw = 2, label='ROC curve', clip_on=False)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve, AUC = %.2f'%auc)
plt.legend(loc="lower right")
plt.savefig('AUC_example.png')
plt.show()

from sklearn.model_selection import GridSearchCV  
from sklearn.linear_model import SGDClassifier 
from sklearn.metrics import roc_curve, auc

# Redefine X data as maxVals and Y data as predicate type with 70/30 test/train split
X_train,X_test,y_train,y_test = train_test_split(maxVals,columns[0:-1],test_size=0.3,random_state=0) 

# Scale X train and test data between -1 and 1 for model training
X_train= np.array(X_train).reshape(-1, 1)
X_test = np.array(X_test).reshape(-1, 1)

# Define stochastic gradient classifier model
model = SGDClassifier(loss='hinge')

# Fit model on training data
model.fit(X_train, y_train)

# Use model to predict training and test Y values based on X values
y_train_pred = model.decision_function(X_train)    
y_test_pred = model.decision_function(X_test) 

# Get False and True Positive rates for training and test predictions
train_fpr, train_tpr, tr_thresholds = roc_curve(y_train, y_train_pred)
test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_test_pred)

# Visualize results with ROC curve
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

# Define Logistic Regression model
log_regression = LogisticRegression()


# Fit model on training data
log_regression.fit(X_train,y_train)

# Use model to predict test Y values based on X values
y_pred_proba = log_regression.predict_proba(X_test)[::,1]

# Get False and True Positive rates for test predictions
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)

# Visualize results with ROC curveplt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# Create 2 dimensional pandas dataframe with maxVals on X and predicate type on Y
df = pd.DataFrame(dict(xPoint=maxVals, yPoint=yPoint, label = label))

# Visualize maxVals as 2 1D numberlines
# Red points at Y=1 is embedding MAV of triples with treats relationship
# Blue points at Y=-1 is embedding MAV of triples with causes relationship
fig, ax = plt.subplots()
colors = {"treats":"red", "causes":"blue"}
ax.scatter(df['xPoint'], df['yPoint'], c=df['label'].map(colors))
plt.show()