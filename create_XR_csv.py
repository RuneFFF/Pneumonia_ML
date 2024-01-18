import csv
import os
#import pandas as pd

def makeCSV(path):
    with open('chest_xray_data.csv','w',newline='') as file:
        writer = csv.writer(file)
        headerList = ['feature', 'name', 'label']
        writer.writerow(headerList)
        for dir, _, files in os.walk(path):
            for f in files:
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
    # #add header
    # headerList = ['type', 'feature', 'name']  # type=test/train/val;feature=NORMAL/PNEUMONIA, name=Image name
    # file = pd.read_csv('chest_xray_data.csv')
    # file.to_csv('chest_xray_data.csv', header=headerList, index=True)

makeCSV('../chest_xray')