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
dataPath = 'C:\\Users\\Aluno\Documents\Python Scripts\Python 2023\cena2-metroind\Original_Data'



namelist = []
# get the name of csv files
for name in os.listdir(dataPath):
    if name.startswith('csv-'):
        namelist.append(name)

#print(namelist)

# create a list with the name of folders
folder_list = [x.replace('.csv', '').replace('csv-cena2-', '') for x in namelist]

folderPath = os.path.join(fileDirectory, 'work_dir')   

folder_list_path = os.path.join(folderPath, 'folderList.csv')
list_df = pd.DataFrame(folder_list)
list_df.to_csv(folder_list_path)

#print(list_df)


for row in range(len(folder_list)):
    path = os.path.join(folderPath, folder_list[row])
    try:
        os.mkdir(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


adj = pd.read_csv('Original_Data\Mat-adj-com-junc.csv', header=None)

adj_df = pd.DataFrame(adj)

# Configura o pandas para exibir todas as colunas sem truncamento
pd.set_option('display.max_columns', None)

# Configura o pandas para exibir todas as linhas sem truncamento
pd.set_option('display.max_rows', None)

#print(adj_df)

'''
N14 - N2
N15 - N14
N16 - N15
N17 - N15
N18 - N14
N19 - N18
N20 - N19

'''


all_node_connections = get_all_connections(adj)

    # Exemplo: Mostre as conexões de todos os nós
for i, connections in enumerate(all_node_connections):
    print(f"Conexões do nó {i}: {connections}")

#sys.exit()

for folders in range(len(namelist)):  

    pa = 20    
    pb = 39
    la = 115
    lb = 133
    
    csvPath = os.path.join(dataPath, namelist[folders])   
    
    df = pd.read_csv(csvPath, header=None, index_col=None).iloc[:, np.r_[pa:pb, la:lb]]

    
    Header_p = ['P2','P3','P4','P5','P6','P7','P8','P9','P10','P11','P12','P13','P14','P15','P16','P17','P18','P19','P20']
    Header_Lb = ['Lb_21', 'Lb_22','Lb_23', 'Lb_24', 'Lb_25', 'Lb_26', 'Lb_27', 'Lb_28', 'Lb_29', 'Lb_30', 'Lb_31', 'Lb_32','Lb_33', 'Lb_34', 'Lb_35', 'Lb_36', 'Lb_37', 'Lb_38']
    Header = Header_p + Header_Lb 

    df.columns=Header

    df_junc = df.copy()
    df_junc[['P14','P15','P16','P17','P18','P19','P20']] = 0

    b = [0.9991, 0.9992, 0.9993, 0.9994, 0.9995, 0.9996, 0.9997]

    df_junc['P14'] = (b[0]*df_junc["P2"]).round(3)
    df_junc['P15'] = (b[1]*df_junc["P14"]).round(3)
    df_junc['P16'] = (b[2]*df_junc["P15"]).round(3)
    df_junc['P17'] = (b[3]*df_junc["P15"]).round(3)
    df_junc['P18'] = (b[4]*df_junc["P14"]).round(3)
    df_junc['P19'] = (b[5]*df_junc["P18"]).round(3)
    df_junc['P20'] = (b[6]*df_junc["P19"]).round(3)


    # Define the folder path
    folder_path = 'C:\\Users\\Aluno\Documents\Python Scripts\Python 2023\cena2-metroind\FilesWreg'
    

    # Define the file name (without the .csv extension)
    file_name = folder_list[folders]
    print(folder_list[folders])

    # Create the full file path by joining the folder path and file name
    full_file_path = os.path.join(folder_path, 'csv-cena2-'+ file_name + '.csv')

    
    df_junc.to_csv(full_file_path, index=False, header=False)

    
