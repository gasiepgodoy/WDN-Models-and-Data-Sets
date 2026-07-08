import numpy as np
import pandas as pd
import torch
import natsort
import os
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import time

torch.manual_seed(101)
np.random.seed(101)

start_time = time.time()


folder = 'Original_Data/cena1-aum-600dias/'       # pasta do dataset -> deve conter a matriz de adjs.
data_name ='Cenas1-aum-600.pt'                   # nome do arquivo pt

a = 'passed-sed'
b = a.split('passed-')
print(b[1])



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



seedlist = []
# verifica as pastas seed
for seed in os.listdir(folder):
    if seed.startswith('csv-'):
        seedlist.append(seed)
        
print('seedlist', seedlist)
print('len_seedlist', len(seedlist))
print('\n')

for folders in range(len(seedlist)):
    #start_time = time.time()    
    print('Currently Seed', seedlist[folders])    
    print('Preproces_Data_Started')
    print('\n')
    
    seed_number = seedlist[folders]
    
    
    s_n = seed_number.split('csv-')[1]
    print('seed_number: ',s_n)
    print('\n')
    
    
    Mcj = pd.read_csv(folder + seedlist[folders] + '/Mat-adj-com-junc.csv', header=None)
    
    
    list1=[]
    
    for x in os.listdir(folder + seed_number):
        if x.startswith('data_'):
            list1.append(x)
    
    srt = natsort.natsorted(list1) # str - data_1, data_2........
    
    #print(srt)
    #print('shape', np.shape(srt))
    
    pa = 9
    pb = 17
    la = 49
    lb = 54
    
    df1 = pd.read_csv(folder + seed_number + '/' + 'data_1.csv', header=None)
    #print(np.shape(df1))
    
    
    df = []
    for filename in srt:
        df.append(pd.read_csv(folder + seed_number + '/' + filename, header=None, index_col=None).iloc[:, np.r_[pa:pb, la:lb]])
        #tm = round(((time.time() - start_time)
        #print('time(s)', round((time.time() - start_time),2))
    
    # concat datasets
    data=df[0]

    
    
    for index in range(1, len(df)):
        data=pd.concat([data, df[index]], axis=0)
        #print('time(s)', round((time.time() - start_time),2))
    
    
    print('concat_data_shape',np.shape(data))
    print('data_0_shape', np.shape(df[0]))
    print('\n')
    #print(np.shape(data))
    Header_p = ['P2','P3','P4','P5','P6','P7','P8','P9']
    Header_Lb = ['Lb_10', 'Lb_11','Lb_12', 'Lb_13', 'Lb_14']
    Header = Header_p + Header_Lb
    
    
    # Define dataframe
    data.columns=Header
    for i in range(len(Header_Lb)):
        print('label', Header_Lb[i])
        print('data_unique',data[Header_Lb[i]].nunique())
        print('data_unique',data[Header_Lb[i]].unique())
        print('data_unique',data[Header_Lb[i]].value_counts()) 
        print('\n')   
    
    
    data['P31'] = data['P3'] - data['P2']
    data['P41'] = data['P4'] - data['P3']
    data['P51'] = data['P5'] - data['P4']
    data['P61'] = data['P6'] - data['P5']
    data['P71'] = data['P7'] - data['P6']
    data['P81'] = data['P8'] - data['P7']
    data['P91'] = data['P9'] - data['P8']

    
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
    
    # Equivalencia:
    # Original / GNN
    # N2 = 0
    # N9 = 1
    # N10 = 2
    # N12 = 3
    # N11 = 4
    # N3 = 5
    # N4 = 6
    # N7 = 7
    # N8 = 8
    # N5 = 9
    # N6 = 10
    
    '''
    data.loc[data['Lb_13'] == 1,  'label'] = 5 
    data.loc[data['Lb_14'] == 1,  'label'] = 6 
    data.loc[data['Lb_15'] == 1,  'label'] = 9 
    data.loc[data['Lb_16'] == 1,  'label'] = 10 
    data.loc[data['Lb_17'] == 1,  'label'] = 7  # dataset 18s
    data.loc[data['Lb_18'] == 1,  'label'] = 8 
    data.loc[data['Lb_19'] == 1,  'label'] = 1 
    data.loc[data['Lb_20'] == 1,  'label'] = 2 
    data.loc[data['Lb_21'] == 1,  'label'] = 4 
    data.loc[data['Lb_22'] == 1,  'label'] = 3 
    '''
    data.loc[data['Lb_10'] == 1,  'label'] = 1
    data.loc[data['Lb_11'] == 1,  'label'] = 4
    data.loc[data['Lb_12'] == 1,  'label'] = 5
    data.loc[data['Lb_13'] == 1,  'label'] = 6
    data.loc[data['Lb_14'] == 1,  'label'] = 7

    
    
    #for (columnName, columnData) in data.iteritems():
    for (RowIndex, rowData) in data[Header_Lb].iterrows():
        #print('Colunm Intex : ', RowIndex)
        #print('Column Contents : ', rowData.values)
        
        if rowData.values.sum() == 0.0:
            data['label'].iloc[RowIndex] = 0
            
        #print('sum', rowData.values.sum())
    '''
    data.loc[(data['Lb_13'] == 0) & (data['Lb_14'] == 0) & (data['Lb_15'] == 0) & (data['Lb_16'] == 0) & (data['Lb_17'] == 0) & (data['Lb_18'] == 0) & \
             (data['Lb_19'] == 0) & (data['Lb_20'] == 0) & \
             (data['Lb_21'] == 0) & (data['Lb_22'] == 0), 'label'] = 0
             #(data['Lb_14'] == 0), 'label'] = 0
    '''        
    
    print('shape data', np.shape(data))
    data_not_duplicate = data.drop_duplicates()
    print('shape data', np.shape(data_not_duplicate))
    print('Original_Data',data.head())
    print('\n')
    print('Not_duplicate_data',data_not_duplicate.head())
    print('not_duplicate_rows (%) =',round(np.shape(data_not_duplicate)[0] / np.shape(data)[0],2))
    print('\n')
    
    
    #mapping={0:'N2',1:'N11',2:'N12',3:'N13',4:'N3',5:'N4',6:'N5',7:'N6',8:'N7',9:'N8',10:'N9',11:'N10'}
    
    data['N0'] = data[['P2']].values.tolist()
    data['N1'] = data[['P7']].values.tolist()
    data['N2'] = data[['P8']].values.tolist()
    data['N3'] = data[['P9']].values.tolist()
    data['N4'] = data[['P5']].values.tolist()
    data['N5'] = data[['P6']].values.tolist()
    data['N6'] = data[['P3']].values.tolist()
    data['N7'] = data[['P4']].values.tolist()

    print('data_unique',data['label'].nunique())
    print('data_unique',data['label'].unique())
    print('data_unique',data['label'].value_counts())
    
    # criando um novo dataset e separando em Data e Label.
    NHeader = ['N0', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7']
    #NHeader = ['N0', 'N1', 'N2', 'N3', 'N4']
    DataN = data[NHeader]
    

    Label = data['label']
    Label = pd.DataFrame(Label)
    
    DataN = DataN.to_numpy()
    Label = Label.to_numpy()
    
    print('value_unique', pd.DataFrame(Label).value_counts())
    print('unique', pd.DataFrame(Label).nunique())
    
    
    x = DataN
    y = Label
            
    Dataset = Tensor_Dataset(x, y)
    
    print('\n')
    print('last_dataset', np.shape(Dataset))
    print('\n')

    #s_n = seed_number.split('csv-')[1]
    
    #torch.save(TrainDataset, folder +'train_dataset15.pt')
    
    Adj_c_j = Dist_to_Adj(Mcj)
    Adj_c_j_tensor = torch.stack((Adj_c_j, Adj_c_j)).type(torch.double)
    
    torch.save(Adj_c_j_tensor, 'torch_data/' + 'adj_matrices.pt')
    save_name = 'torch_data/' + s_n + data_name
    torch.save(Dataset, save_name)
    print('Save_name:', save_name) 
    print('\n')
    
    
print('Preproces_Data_Finished for seed {} ------ Total_time(s): {} '.format( s_n ,round((time.time() - start_time),2)))



