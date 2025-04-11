## CENA 1

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import sys
import os
import errno

np.random.seed(101)

def get_all_connections(adj_matrix):
    num_nodes = len(adj_matrix)
    all_connections = []

    for i in range(num_nodes):
        connections = []
        for j, connected in enumerate(adj_matrix[i]):
            if connected != 0:
                connections.append(j)
        all_connections.append(connections)

    return all_connections


# Paths
absolutepath = os.path.abspath(__file__)
print('ab',absolutepath)
fileDirectory = os.path.dirname(absolutepath)
print('file', fileDirectory)
parentDirectory = os.path.dirname(fileDirectory)
print('parent', parentDirectory)
dataPath = 'C:\\Users\\Aluno\Documents\Python Scripts\Python 2023\cena 1\Original_Data'


namelist = []
# get the name of csv files
for name in os.listdir(dataPath):
    if name.startswith('csv-'):
        namelist.append(name)
#
# create a list with the name of folders
folder_list = [x.replace('.csv', '').replace('csv-cena1-', '') for x in namelist]

folderPath = os.path.join(fileDirectory, 'work_dir')   

folder_list_path = os.path.join(folderPath, 'folderList.csv')
list_df = pd.DataFrame(folder_list)
list_df.to_csv(folder_list_path)

for row in range(len(folder_list)):
    path = os.path.join(folderPath, folder_list[row])
    try:
        os.mkdir(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


adj = pd.read_csv('.\Original_Data\Mat-adj-com-junc.csv', header=None)

adj_df = pd.DataFrame(adj)

# Configura o pandas para exibir todas as colunas sem truncamento
pd.set_option('display.max_columns', None)

# Configura o pandas para exibir todas as linhas sem truncamento
pd.set_option('display.max_rows', None)
















all_node_connections = get_all_connections(adj)

    # Exemplo: Mostre as conexões de todos os nós
for i, connections in enumerate(all_node_connections):
    print(f"Conexões do nó {i}: {connections}")

for folders in range(len(namelist)):  

    pa = 9
    pb = 17
    la = 49
    lb = 54
    
    csvPath = os.path.join(dataPath, namelist[folders])   
    
    df = pd.read_csv(csvPath, header=None, index_col=None).iloc[:, np.r_[pa:pb, la:lb]]

    
    Header_p = ['P2','P3','P4','P5','P6','P7','P8','P9']
    Header_Lb = ['Lb_10', 'Lb_11','Lb_12', 'Lb_13', 'Lb_14']
    Header = Header_p + Header_Lb

    df.columns=Header

    df_junc = df.copy()
    df_junc[['P7','P8','P9']] = 0

    b = [0.9991, 0.9992, 0.9993]

    df_junc['P7'] = (b[0]*df_junc["P2"]).round(3)
    df_junc['P8'] = (b[1]*df_junc["P7"]).round(3)
    df_junc['P9'] = (b[2]*df_junc["P8"]).round(3)

    # Define the folder path
    folder_path = 'C:\\Users\\Aluno\Documents\Python Scripts\Python 2023\cena 1\FilesWreg'

    # Define the file name (without the .csv extension)
    file_name = folder_list[folders]
    print(folder_list[folders])
    # Create the full file path by joining the folder path and file name
   # full_file_path = os.path.join(folder_path, file_name + '.csv')
    full_file_path = os.path.join(folder_path, 'csv-cena1-'+ file_name + '.csv')
    
    df_junc.to_csv(full_file_path, index=False, header=False)
