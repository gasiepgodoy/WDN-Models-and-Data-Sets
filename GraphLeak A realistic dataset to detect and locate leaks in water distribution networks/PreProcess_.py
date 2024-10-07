import numpy as np
import pandas as pd
import torch
import os
from torch.utils.data import TensorDataset
from sklearn import preprocessing
import time
import sys
import re
import argparse

parser = argparse.ArgumentParser(description='GraphLeak')

parser.add_argument('--data_folder', type=str, default='Original_Data/Dataset14/WDN1/csv-dist-orig')
parser.add_argument('--save_folder', type=str, default='work_dir/')

parser.add_argument('--n_nodes', type=int, default=8)
parser.add_argument('--n_leaks', type=int, default=5)

parser.add_argument('--pressure', type=bool, default=True, help='Pressure?')
parser.add_argument('--flow', type=bool, default=True, help='Flow?')
parser.add_argument('--volume', type=bool, default=False, help='Volume?')

parser.add_argument('--Noise', type=bool, default=False, help='Do you want to Gaussian noise in their data?')
parser.add_argument('--mu', type=float, default=0.0, help='If you chose Gaussian noise, whats is the mean value?')
parser.add_argument('--sigma', type=float, default=0.1, help='If you chose Gaussian noise, whats is the standard deviation value?')

parser.add_argument('--Nodes_normalization', type=bool, default=True, help='Do you want normalize the values between nodes?')
parser.add_argument('--Data_Normalization', type=bool, default=True, help='Do you want normalize the values on the range 0 to 1?')

args = parser.parse_args()

# random seeds
torch.manual_seed(101)
np.random.seed(101)

start_time = time.time()

# set GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_folder = args.data_folder
save_folder = args.save_folder
pressure = args.pressure
flow = args.flow
volume = args.volume
noise = args.Noise
mu = args.mu
sigma = args.sigma
nodes_normalization = args.Nodes_normalization
data_normalization = args.Data_Normalization
n_nodes = args.n_nodes
n_leaks = args.n_leaks

torch.manual_seed(101)
np.random.seed(101)

start_time = time.time()

def create_tensor_dataset(x, y):
    tensor_x = torch.Tensor(x).double()
    tensor_y = torch.LongTensor([item.item() for item in y])
    tensor_z = torch.tensor(1, dtype=torch.int32)
    tensor_A = torch.LongTensor([tensor_z] * len(x))

    my_dataset = TensorDataset(tensor_x, tensor_y, tensor_A)

    return my_dataset

def Dist_to_Adj(Adj):
    Adj1 = (Adj >= 1).astype(int)
    return Adj1

def normalize_data(data):
    x = data.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    normalized_data = pd.DataFrame(x_scaled)
    
    return normalized_data

def add_noise(data, column_name, mu=mu, sigma=sigma):
    d1 = data[column_name]
    gauss_dimension = np.shape(d1)

    noise = np.random.normal(mu, sigma, gauss_dimension)
    
    data_with_noise = d1 + noise
    
    data[column_name] = data_with_noise
    
    return data

def node_normalization(data, columns_to_difference, reference_column):
    for col in columns_to_difference:
        data[col] = data[col] - data[reference_column]
    
    return data

def get_measurement_values(data_folder, flow, pressure, volume, n_nodes=n_nodes, n_leaks=n_leaks):
    df1 = pd.read_csv(data_folder + '/data_1.csv', header=None)


    fl_in, fl_out = 1, (1 + n_nodes)
    pr_in, pr_out = (1 + n_nodes), (n_nodes * 2 + 1)
    vo_in, vo_out = (n_nodes * 2 + 1), (n_nodes * 3 + 1)
    coo_in, coo_out = (n_nodes * 3 + 1), (n_nodes * 6 + 1)
    l_in, l_out = (n_nodes * 6 + 1), (n_nodes * 6 + 1) + 5
    lcoo_in, lcoo_out = (n_nodes * 6 + 1) + 5, (n_nodes * 6 + 1) + 3 + 5
    wd_in, wd_out = (n_nodes * 6 + 1) + 3 + 5, n_nodes * 6 + 1

    df_measurements = pd.DataFrame()

    pa1, pa2, pa3 = float('inf'), float('inf'), float('inf')
    pb1, pb2, pb3 = float('-inf'), float('-inf'), float('-inf')

    header_F, header_P, header_V = [], [], []  # Define empty lists

    if flow:
        df_measurements = pd.concat([df_measurements, df1.iloc[:, fl_in:fl_out]], axis=1)
        pa1, pb1 = fl_in, fl_out
        header_F = ['F' + str(i) for i in range(1, n_nodes + 1)]
    if pressure:
        df_measurements = pd.concat([df_measurements, df1.iloc[:, pr_in:pr_out]], axis=1)
        pa2, pb2 = pr_in, pr_out
        header_P = ['P' + str(i) for i in range(1, n_nodes + 1)]
    if volume:
        df_measurements = pd.concat([df_measurements, df1.iloc[:, vo_in:vo_out]], axis=1)
        pa3, pb3 = vo_in, vo_out
        header_V = ['V' + str(i) for i in range(1, n_nodes + 1)]

    pa_ = min(pa1, pa2, pa3)
    pb_ = max(pb1, pb2, pb3)

    header_p = header_F + header_P + header_V if flow or pressure or volume else []
    header_l = ['Lb_' + str(i) for i in range(1, n_leaks + 1)]

    return l_in, l_out, pa_, pb_, header_p, header_l

def generate_label_mappings(header_Lb):
    label_mappings = {header: index + 1 for index, header in enumerate(header_Lb)}
    return label_mappings

data_folder='cena1\Original_Data\Dataset14\WDN1\csv-dist-orig/'
save_folder='cena1\torch_data\WD1'

data_list = [seed for seed in os.listdir(data_folder) if seed.startswith('data_')]

mcj = pd.read_csv('cena1\Original_Data\Dataset14\WDN1\Mat-adj-com-junc.csv', header=None)

Adj = Dist_to_Adj(mcj)
#Adj_tensor = torch.stack((Adj, Adj)).type(torch.double) # used when a double tensor is required

torch.save(Adj, os.path.join(save_folder, 'adj_matrices.pt'))

sorted_list = sorted(data_list, key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])

for files in range(len(sorted_list)):
    print('Currently file : ', sorted_list[files])

    la, lb, pa, pb, header_p, header_Lb = get_measurement_values(data_folder, flow, pressure, volume, n_nodes, n_leaks)

    df = []
    for filename in sorted_list:
        df.append(pd.read_csv(os.path.join(data_folder, filename), header=None, index_col=None).iloc[:, np.r_[pa:pb, la:lb]])
    
    data = pd.concat(df, axis=0)

    Header = header_p + header_Lb

    data.columns = Header    

    if data_normalization:
        # data normalization
        data = node_normalization(data, header_p[1:], header_p[0])
    
    if nodes_normalization:
        # node normalization
        data = normalize_data(data)
        data.columns = Header

    if noise:
        data=add_noise(data, header_p, mu=mu, sigma=sigma)

    # labels

    #label_mappings = {'Lb_10': 1, 'Lb_11': 4, 'Lb_12': 5, 'Lb_13': 6, 'Lb_14': 7}
    label_mappings = generate_label_mappings(header_Lb)
    
    data['label'] = data[header_Lb].idxmax(axis=1).map(label_mappings)#.fillna(0)
    data.loc[data[header_Lb].sum(axis=1) == 0, 'label'] = 0
    
    '''
    idxmax(axis=1) is called on the subset DataFrame. 
    It finds the column name with the maximum value for each row along the horizontal axis (axis=1). 
    This identifies the column with the highest value among the selected columns for each row.

    The map(label_mappings) method is then called on the result of idxmax(axis=1). 
    It maps the column names to their corresponding label values using the label_mappings dictionary. 
    For example, if the maximum value column is 'Lb_10', it will be mapped to 1.
    
    '''
    adj_matrices = Dist_to_Adj(mcj)

    x = data[header_p].to_numpy()
    y = data['label'].to_numpy()

    my_dataset = create_tensor_dataset(x, y)

    data_name=sorted_list[files].split('.csv')[0]

    SavePath = os.path.join(save_folder, data_name) 

    torch.save(my_dataset, SavePath)

    print('SavePath=',SavePath)
   
    print('Step {}/{} ----------> file {} converted from csv to torch data'.
          format(files, len(sorted_list), sorted_list[files]))
     
print('Total_time: {} s '.format(round((time.time() - start_time),2)+1))