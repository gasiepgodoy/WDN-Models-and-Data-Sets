import numpy as np
import pandas as pd
import torch
import natsort
import os
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import time
import argparse
import sys
from pathlib import Path
import errno

torch.manual_seed(101)
np.random.seed(101)

start_time = time.time()

def Norm(Data):
    x = Data.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    Data = pd.DataFrame(x_scaled)
    
    return Data



def Tensor_Dataset(x, y):

    # z = Como o algoritmo trabalha com multiplas topologias e o nosso tem apenas uma, 
    # criei um tensor que linka todos os dados a nossa matriz adjacência


    e = []
    z = torch.tensor(1, dtype=torch.int32) 
    A = []
    for a in range(np.shape(x)[0]):
        d = []
        for b in range(np.shape(x)[1]):
            c = np.array(x[a][b])        
            d.append(c)
            #print('d',d)
        e.append(d)
        A.append(z)
    
    y_ = []
    for i in y:
        #y_.append(y[i].item())
        y_.append(i.item())
        
    e1 = torch.Tensor(e).type(torch.double)   # X
    f1 = torch.LongTensor(y_)                 # Label
    A1 = torch.LongTensor(A)                  # Localização matriz Adjacência
    
    
    my_dataset = TensorDataset(e1, f1, A1)

    # my_dataset[0] - [X, Y, A]

    return my_dataset


def Dist_to_Adj(Adj):
    for i in range(len(Adj)):
        for j in range(len(Adj)):
            if Adj[i][j] >= 1:
                Adj[i][j] = 1

    Adj1 = np.array(Adj)
    adj_matrices = torch.from_numpy(Adj1)

    return adj_matrices


import os
import sys

# Paths
absolutepath = os.path.abspath(__file__)
print('ab',absolutepath)
fileDirectory = os.path.dirname(absolutepath)
print('file', fileDirectory)
parentDirectory = os.path.dirname(fileDirectory)
print('parent', parentDirectory)
dataPath = os.path.join(fileDirectory, '.\Original_Data')   
#

namelist = []
# get the name of csv files
for name in os.listdir(dataPath):
    if name.startswith('csv-'):
        namelist.append(name)
#
# create a list with the name of folders
folder_list = [x.replace('.csv', '').replace('csv-cena1-', '') for x in namelist]

folderPath = os.path.join(fileDirectory, './work_dir')   
print('folder', folderPath)
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

for folders in range(len(namelist)):   
    
    Mcj = pd.read_csv(dataPath + '/Mat-adj-com-junc.csv', header=None)
    
 
    #list1=[]
    
    
    pa = 9
    pb = 17
    la = 49
    lb = 54
    
    csvPath = os.path.join(dataPath, namelist[folders])   
    
    df = pd.read_csv(csvPath, header=None, index_col=None).iloc[:, np.r_[pa:pb, la:lb]]
    
    Header_p = ['P2','P3','P4','P5','P6','P7','P8','P9']
    Header_Lb = ['Lb_10', 'Lb_11','Lb_12', 'Lb_13', 'Lb_14']
    Header = Header_p + Header_Lb  

    data = df
    data.columns=Header  

    data['P31'] = data['P3'] - data['P2']
    data['P41'] = data['P4'] - data['P2']
    data['P51'] = data['P5'] - data['P2']
    data['P61'] = data['P6'] - data['P2']
    data['P71'] = data['P7'] - data['P2']
    data['P81'] = data['P8'] - data['P2']
    data['P91'] = data['P9'] - data['P2']

    
    data['P3'] = data['P31']
    data['P4'] = data['P41']
    data['P5'] = data['P51']
    data['P6'] = data['P61']
    data['P7'] = data['P71']
    data['P8'] = data['P81']
    data['P9'] = data['P91']

    
    data = data.drop('P31', axis = 1)
    data = data.drop('P41', axis = 1)
    data = data.drop('P51', axis = 1)
    data = data.drop('P61', axis = 1)
    data = data.drop('P71', axis = 1)
    data = data.drop('P81', axis = 1)
    data = data.drop('P91', axis = 1)

    ############# Noise
    '''
    print('Original_Data')
    print(data.head())
    
    d1 = data[Header_p]
    gauss_dimension=np.shape(d1)
    
    mu, sigma = 0, 0.1
    noise = np.random.normal(mu, sigma, gauss_dimension)
    print(np.shape(noise))
    
    data_with_noise = d1 + noise
    
    print('dnoise',data_with_noise.head())
    
    data[Header_p] = data_with_noise
    print('data_noise', data.head())
    '''
    data = Norm(data)
    
    data.columns = Header
    
    data.loc[data['Lb_10'] == 1,  'label'] = 1
    data.loc[data['Lb_11'] == 1,  'label'] = 4
    data.loc[data['Lb_12'] == 1,  'label'] = 5
    data.loc[data['Lb_13'] == 1,  'label'] = 6
    data.loc[data['Lb_14'] == 1,  'label'] = 7
 
    for (RowIndex, rowData) in data[Header_Lb].iterrows():

        if rowData.values.sum() == 0.0:
            data['label'].iloc[RowIndex] = 0
            
    data_not_duplicate = data.drop_duplicates()

    data['N0'] = data[['P2']].values.tolist()
    data['N1'] = data[['P3']].values.tolist()
    data['N2'] = data[['P4']].values.tolist()
    data['N3'] = data[['P5']].values.tolist()
    data['N4'] = data[['P6']].values.tolist()
    data['N5'] = data[['P7']].values.tolist()
    data['N6'] = data[['P8']].values.tolist()
    data['N7'] = data[['P9']].values.tolist()

    NHeader = ['N0', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7']
    #NHeader = ['N0', 'N1', 'N2', 'N3', 'N4']
    DataN = data[NHeader]
    

    Label = data['label']
    Label = pd.DataFrame(Label)
    
    DataN = DataN.to_numpy()
    Label = Label.to_numpy()
  
    
    x = DataN
    y = Label
            
    Dataset = Tensor_Dataset(x, y)
    
    Adj_c_j = Dist_to_Adj(Mcj)
    Adj_c_j_tensor = torch.stack((Adj_c_j, Adj_c_j)).type(torch.double)

    SavePath = os.path.join(folderPath,folder_list[folders] )   
    
    torch.save(Adj_c_j_tensor, os.path.join(SavePath, 'adj_matrices.pt'))
    torch.save(Dataset,os.path.join(SavePath, 'dataset.pt'))
    
    print('Step {}/{} ----------> file {} converted from csv to torch data'.format(folders, len(namelist), namelist[folders]))
    
    
print('Total_time: {} s '.format(round((time.time() - start_time),2)+1))



