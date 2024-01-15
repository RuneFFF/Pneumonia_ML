import csv
import os
import pandas as pd

def makeCSV(path):
    with open('chest_xray_data.csv','w',newline='') as file:
        writer = csv.writer(file)
        headerList = ['type', 'feature', 'name']  # type=test/train/val;feature=NORMAL/PNEUMONIA, name=Image name
        writer.writerow(headerList)
        for dir, _, files in os.walk(path):
            for f in files:
                if f[-5:] == ('.jpeg'):
                    #get type directory
                    if 'test' in dir:
                        type = 'test'
                    elif 'train' in dir:
                        type = 'train'
                    else:
                        type = 'val'

                    #get feature directory
                    if 'NORMAL' in dir:
                        feature = 'NORMAL'
                    else:
                        feature = 'PNEUMONIA'
                    file_name = f.split('.')[0]
                    writer.writerow([type, feature, file_name])
    # #add header
    # headerList = ['type', 'feature', 'name']  # type=test/train/val;feature=NORMAL/PNEUMONIA, name=Image name
    # file = pd.read_csv('chest_xray_data.csv')
    # file.to_csv('chest_xray_data.csv', header=headerList, index=True)