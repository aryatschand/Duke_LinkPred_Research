import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


data = {}
file1 = open('data.txt', 'r')
lines = file1.readlines()

for line in lines:
    lineSplit = line.split(" ")
    data[lineSplit[3].replace("\n", "")] = lineSplit[2]
file1.close()

file1 = open('vector.txt', 'r')
lines = file1.readlines()
 
count = 0
first = True

xPoint = []
yPoint = []
label = []

# Strips the newline character
for line in lines:
    count += 1

    if first:
        first = False
    else:
        lineSplit = line.split(" ")
        label.append(data[lineSplit[0]])
        xPoint.append(float(lineSplit[1]))
        yPoint.append(float(lineSplit[2]))

df = pd.DataFrame(dict(xPoint=xPoint, yPoint=yPoint, label = label))

fig, ax = plt.subplots()

colors = {"1":"red", "0":"blue"}

ax.scatter(df['xPoint'], df['yPoint'], c=df['label'].map(colors))

plt.show()