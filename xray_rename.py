import os
import numpy as np

def x_rename(path):
    n_normal = 0
    n_virus = 0
    n_bacteria = 0
    #direct = os.listdir(path)
    for dir,_, files in os.walk(path):
        for f in files:
            if f[-5:] == ('.jpeg'):
                file_name = f.split('.')[0]
                ext = f.split('.')[-1]
                if 'virus' in file_name or 'v' in file_name:
                    person_number = file_name.split('_')[0].replace('person','')
                    new_name = 'p_'+person_number+'_v_'+str(n_virus)+"."+ext
                    n_virus +=1
                elif 'bacteria' in file_name or 'b' in file_name:
                    person_number = file_name.split('_')[0].replace('person', '')
                    new_name = 'p_'+person_number+'_b_'+str(n_bacteria)+"."+ext
                    n_bacteria +=1
                else:
                    new_name = 'n_'+str(n_normal)+"."+ext
                    n_normal +=1

                os.rename(os.path.join(dir, f), os.path.join(dir, new_name))

#x_rename('chest_xray')