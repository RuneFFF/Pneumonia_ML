import csv
import os
import random

def makeCSV(path):
    with open('chest_xray_data.csv','w',newline='') as file:
        writer = csv.writer(file)
        headerList = ['feature', 'name', 'label']
        writer.writerow(headerList)
        dir = os.listdir(path)
        for f in random.sample(dir, len(dir)):
                if f[-5:] == ('.jpeg'):
                    #get feature
                    if 'n' in f:
                        feature = 'NORMAL'
                        label = 'N'
                    else:
                        feature = 'PNEUMONIA'
                        if 'b' in f:
                            label = 'B'
                        elif 'v' in f:
                            label = 'V'
                    file_name = f.split('.')[0]
                    writer.writerow([feature, file_name, label])

makeCSV('../chest_xray')