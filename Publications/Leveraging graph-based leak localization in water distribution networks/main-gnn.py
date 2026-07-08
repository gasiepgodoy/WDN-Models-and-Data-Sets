'''
19/04/23
- Adicionado break de acordo com o valor adicionado em 'limit'
- Corrigido erro no nome dos arquivos

07/05/23
- Adicionado contador e controle dos testes.

'''

from sklearn.preprocessing import Normalizer
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from model import GGNN
import torch.nn as nn
import time
import sys
from tqdm import tqdm
import os
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import random
import json
import pickle

random.seed(101)

# time
now = datetime.now()
d_print = now.strftime("%d%m_%H%M")
print(d_print)

# set GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

folder='torch_data/'
Save_folder ='work_dir/'

parser = argparse.ArgumentParser(description='linkage indetification')
parser.add_argument('--n_epochs', type=int, default=220, help='Number of Epochs') #300
parser.add_argument('--windows_size', type=int, default=5, help='Neighborhood Interation')
parser.add_argument('--hidden_layer_size', type=int, default=256, help='Number of Neurons in Hidden Layer')
parser.add_argument('--batch_size', type=int, default=300, help='Batch size')  #300
parser.add_argument('--Train_dataset', type=str, default='seed10Cenas.pt', help='Dataset with linkage mensurements in .pt format')
parser.add_argument('--Train_adj', type=str, default='adj_matrices.pt', help='Adjacency Matrix of Train Datast')
parser.add_argument('--Val_dataset', type=str, default='seed101Cenas.pt', help='Dataset with linkage mensurements in .pt format')
parser.add_argument('--Val_adj', type=str, default='adj_matrices.pt', help='Adjacency Matrix of Test Datast')
parser.add_argument('--k', type=int, default=10, help='number of k-folds') #10
parser.add_argument('--Phase', type=str, default='train', help='Train')
parser.add_argument('--limit', type=int, default=10000, help='Trainning number')
parser.add_argument('--timeControl', type=int, default=5, help='Time control')
parser.add_argument('--distControl', type=int, default=5, help='Distance control')



args = parser.parse_args()

batch_size = args.batch_size
num_epochs = args.n_epochs
k = args.k
Val_adj = args.Val_adj
Val_dataset = args.Val_dataset
Train_dataset = args.Train_dataset
Train_adj = args.Train_adj
hidden_layer = args.hidden_layer_size
windows_size = args.windows_size
Phase = args.Phase

Limit = args.limit
DistanceControl = args.distControl
TimeControl = args.timeControl

# Paths
absolutepath = os.path.abspath(__file__)
print('absolutepath=',absolutepath) 
fileDirectory = os.path.dirname(absolutepath)
print('fileDirectory=', fileDirectory) 
parentDirectory = os.path.dirname(fileDirectory)
print('parentDirectory=', parentDirectory) 

folderPath = os.path.join(fileDirectory, 'work_dir') 
folderPath1 = os.path.join(fileDirectory, 'work_dir2') 

folder_list_path = os.path.join(folderPath, 'folderList.csv')

folders = pd.read_csv(folder_list_path)

model = GGNN.GGNNModel(1, args.hidden_layer_size, windows_size).to(device)
print('Model')
print(model) 


#### k-fold ####

splits=KFold(n_splits=k,shuffle=True,random_state=42)
foldperf={}                                                 # dictionary to save statistics
folderconf={}
################

torch.set_printoptions(threshold =10000)

Y = 0    
def train_epoch(model, device, trainLoader, loss_fn, optimizer, batch_size, adj_matrices):    
    model.train()
 
    train_pred = []
    train_correct = []
    count=0
    for epoch in range(num_epochs):
        train_loss = train_acc = 0.0
        
        X, label, idx = next(iter(trainLoader))

        A = adj_matrices[idx]

        Y = model(X.cuda(), A.cuda())                                 # Result is a weigthed vector given by softmax


        #print('X', X)

        #sys.exit()
        

        predictions = torch.exp(Y).argmax(-1).cpu()
        
        ### mostrar os valores das predições
        
        percentage = torch.nn.functional.softmax(Y, dim=1)*100
        

        #if epoch == (num_epochs-1):
         #   for idx in range(len(Y)):
          #      print(predictions[idx].item(), percentage[idx], label[idx].item()
              
        ###

        loss = criterion(Y, label.cuda())                             # Compute loss
        
        optimizer.zero_grad()                                         # Clear the gradients

        loss.backward()                                               # Calcule gradients
        optimizer.step()                                              # Update Weights

        train_loss += loss.item()                                     # Calculate Loss

        train_acc += ((predictions == label).sum())/len(label)    # Accuracy
        
        train_pred.extend(predictions)
        train_correct.extend(label)
        
        return train_loss, train_acc, train_pred, train_correct, predictions, percentage, label
        
def validation_epoch(model, device, val_loader, loss_fn, adj_matrices):
    
    model.eval()
    
    val_pred = []
    val_correct = []
    counter = 0.0
    val_loss = val_acc = 0.0
    
    for batch in val_loader:
        #print(batch)
        
        X, label, idx = batch                               # batch interation
        A = adj_matrices[idx]                               # Adj Matrices
        Y = model(X.cuda(), A.cuda())                       # Foward Pass
        counter +=1
        loss = loss_fn(Y, label.cuda())                     # Compute loss
        val_loss += loss.item()                             # calculate Loss
        
        predictions = torch.exp(Y).argmax(-1).cpu()    
        
        val_loss += loss.item()                             # Calculate Loss

        val_acc += ((predictions == label).sum())/len(label)    # Accuracy    
        
        val_pred.extend(predictions)
        val_correct.extend(label)

    
        return val_loss, val_acc, val_pred, val_correct
    
def CONTROL(T1,TEMPO_AQ, COUTER):
    c=0
    tempo = T1.split('min')[0]
    idx=TEMPO_AQ.index(T1)
    #print(idx)

    if COUTER[idx] != 0:
        COUTER[idx] -=1
    else:
        print('The limit for time {}mins has been reached'.format(tempo))
        c=-1

    
    return c, COUTER

def D_CONTROL(X, CONTADOR):
    k=0
    D_NAME = ['x1', 'x5', 'x10', 'x25', 'x50', 'x100', 'x200']
    indice = D_NAME.index(X)
    
    if CONTADOR[indice] != 0:
        CONTADOR[indice] -=1
    else:
        print(f'Distance {X} has reached its trainning limit')
        
        k =-1

    return k, CONTADOR



folders.columns = ['n', 'names']
folders = folders.drop('n',axis=1).values.tolist()



df1 = pd.DataFrame(folders)
#df1 = df1.sample(frac=1, random_state=103)
#df2 = df1.sample(frac=1, random_state=101)
#print(df2.head(1))


#folders = df1.values.tolist()
#folders2 = df2.values.tolist()

RESULT = {'train':[], 'val':[], 'results':[]} # Para salvar os valores do history

count = 0

TEMPO_AQ = ['10min','30min','60min','180min','360min','720min']


d=DistanceControl
t=TimeControl


D_NAME = ['x1', 'x5', 'x10', 'x25', 'x50', 'x100', 'x200']
D_COUNTER = [d, d, d, d, d, d, d]  #editei
#COUTER = [t, t, t, t, t, t]
# COUNT10, COUNT30, COUNT60, COUNT180, COUNT360, COUNT720 = 0, 0, 0, 0, 0, 0
if __name__ == '__main__':

    for idx in range(len(D_NAME)):
        DC = D_NAME[idx]
        print('DC',DC)

        #x1, x5, x10, x25, x50, x100
        folder0 = 0
        COUTER = [t, t, t, t, t, t]


        df1 = df1.sample(frac=1, random_state=103)
        df2 = df1.sample(frac=1, random_state=101)


        folders = df1.values.tolist()
        folders2 = df2.values.tolist()

        for folder in range(len(folders)):
            #COUTER = [t, t, t, t, t, t]

            DIS1=folders[folder][0].split('-')[1]
            
            if DIS1 != DC:
                continue
            '''
            k, CONTADOOR=D_CONTROL(DIS1, D_COUNTER)

            if k == -1:
                print('keRROr')
                continue
            
            print(f'CONTROL DISTANCE VECTOR = {CONTADOOR} -->  [x1, x5, x10, x25, x50, x100, x200]')
            '''
            for folder0 in range(len(folders2)):

                

                if Limit<0:
                    #print('\n <<< Limite de treinamentos excedido >>> ')
                    break

                #Limit-=1
                #print('T_STEP {} - V_STEP {}'.format(folder, folder0))
            
                if folder0 == len(folders):
                    print('FINISH !!!')
                    break       
                    
                    #if sum(COUTER) > 0:
                     #   continue
                #else:
                    #print('FINISH !!!')
                    #break

                DIS1=folders[folder][0].split('-')[1]
                DIS2=folders2[folder0][0].split('-')[1]

                T1=folders[folder][0].split('-')[-1]
                T2=folders2[folder0][0].split('-')[-1]
                
                '''
                if DIS1 == 'x1':
                    pass
                else:
                    break
                '''
                
                if DIS1 == DIS2 and T1 == T2:
                    print('\n')
                    print('Train_file=', folders[folder][0])
                    print('Test_file=', folders2[folder0][0])
                    Limit-=1
                    print('Remaining= ', Limit) 
                else:
                    
                    #print('{} and {} are differents -- step {}.'.format(folders[folder][0], folders2[folder0][0], folder0))
                    if folder0 <=len(folders2):
                        folder0+=1
                        continue
                
                
                c, contador = CONTROL(T1,TEMPO_AQ, COUTER)
                if c == -1:
                    break


                print(f'CONTROL VECTOR = {COUTER} PARA DISTÂNCIA = {DIS1}')
                #continue

                #sys.exit()

                tr = folders[folder][0]
                ts  = folders2[folder0][0]
                
                train_path = os.path.join(folderPath, folders[folder][0])
                val_path = os.path.join(folderPath, folders2[folder0][0])
                TrainDataset = torch.load(os.path.join(train_path, 'dataset.pt'))
                ValDataset = torch.load(os.path.join(val_path, 'dataset.pt'))

                #print(TrainDataset)

                #sys.exit()

                #folders[folder][0], folders2[folder0][0])
                
                Ss = folders[folder][0] + folders2[folder0][0]
            
                adjacency = torch.load(os.path.join(train_path,'adj_matrices.pt'))

                #### k-fold ####
                dataset=ConcatDataset([TrainDataset, ValDataset])
                

                for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(dataset)))):

                    train_sample = SubsetRandomSampler(train_idx)
                    val_sample = SubsetRandomSampler(val_idx)
                    
                    train_loader = DataLoader(dataset,batch_size=batch_size, sampler=train_sample)

                    # separar de acordo com as especificações

                    #print('trainloader',train_loader)


                    #sys.exit()



                    val_loader = DataLoader(dataset,batch_size=batch_size, sampler=val_sample)   
                    
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    
                    criterion = nn.CrossEntropyLoss()
                    optimizer = torch.optim.Adam(model.parameters())
                    
                    history = {'train_loss': [], 'val_loss': [],'train_acc':[],'val_acc':[]}
                    
                    data_conf = {'train_pred': [], 'train_corr': [], 'val_pred': [], 'val_corr':[]}
                    
                    
                    for epoch in range(num_epochs):
                        train_loss, train_acc, train_pred, train_label, predictions, percentage, label = train_epoch(model, device, train_loader, criterion, optimizer, batch_size, adjacency)
                        val_loss, val_acc, val_pred, val_label = validation_epoch(model, device, val_loader,criterion, adjacency)
                        
                        
                        train_loss = train_loss
                        val_loss = val_loss
                        train_acc = train_acc*100
                        val_acc = val_acc*100
                                
            
                        history['train_loss'].append(train_loss) # valores de cada k.
                        history['val_loss'].append(val_loss)
                        history['train_acc'].append(train_acc.cpu().detach().numpy())
                        history['val_acc'].append(val_acc.cpu().detach().numpy())
                        
                        data_conf['train_pred'].append(train_pred)
                        data_conf['train_corr'].append(train_label)
                        data_conf['val_pred'].append(val_pred)
                        data_conf['val_corr'].append(val_label)
                        
                        #print(epoch)
                        
                        #if epoch == (num_epochs-1):
                        #  for idx in range(len(label)):
                        #     print(predictions[idx].item(), percentage[idx], label[idx].item())

                    
                    foldperf['fold{}'.format(fold+1)] = history
                    folderconf['fold{}'.format(fold+1)] = data_conf
                    
                    
                    print("Step: {}/{} Fold:{} Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f} AVG Training Acc {:.2f} % AVG Test Acc {:.2f} %".format(folder0+1, len(folders), fold+1,epoch+1, num_epochs, 
                                                                                round(train_loss,3),
                                                                                round(val_loss,3),
                                                                                train_acc,
                                                                                val_acc))
                    
                
                # calculate the average score in every fold & average score of all folds
                
                sc_tl, sc_ta, sc_vl, sc_va = [], [], [], []
                
                nf=args.k # number of folds
                
                torch.save(data_conf, 'work_dir/data_to_Mconf.pt')
                
                for f in range(1,nf+1):
                    sc_tl.append(np.mean(foldperf['fold{}'.format(f)]['train_loss']))
                    sc_ta.append(np.mean(foldperf['fold{}'.format(f)]['val_loss']))
                    
                    sc_vl.append(np.mean(foldperf['fold{}'.format(f)]['train_acc']))
                    sc_va.append(np.mean(foldperf['fold{}'.format(f)]['val_acc']))
                
                print('TR=', tr)
                print('\n')
                RESULT['train'].append(tr)
                RESULT['val'].append(ts)
                RESULT['results'].append(history)       

                

                
                diz_ep = {'train_loss_ep':[],'test_loss_ep':[],'train_acc_ep':[],'test_acc_ep':[]}
                
                for i in range(num_epochs):
                    diz_ep['train_loss_ep'].append(np.mean([foldperf['fold{}'.format(f+1)]['train_loss'][i] for f in range(args.k)]))
                    diz_ep['test_loss_ep'].append(np.mean([foldperf['fold{}'.format(f+1)]['val_loss'][i] for f in range(args.k)]))
                    diz_ep['train_acc_ep'].append(np.mean([foldperf['fold{}'.format(f+1)]['train_acc'][i] for f in range(args.k)]))
                    diz_ep['test_acc_ep'].append(np.mean([foldperf['fold{}'.format(f+1)]['val_acc'][i] for f in range(args.k)]))    


                modelParameters = Ss
                modelName = modelParameters + '_model.pt'
                StatsName = modelParameters + '_training_statistics.pt'
                LossName = modelParameters + '_loss.pt'
                CheckPointName = modelParameters + 'checkpoint.pth'
                
            
                print('Finish process: TRAIN_DATASET: {}, VAL_DATASET:{}'.format(folders[folder][0], folders2[folder0][0]))
                
                checkpoint = {
                'epoch': num_epochs,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict()
                }
                
                #train_path = os.path.join(folderPath, folders[folder][0])

                SavePath = os.path.join(folderPath1, CheckPointName)
                SavePathLoss = os.path.join(folderPath1, LossName)

                torch.save(checkpoint, SavePath)
                torch.save(diz_ep, SavePathLoss)
                
        
                
                #time.sleep(30)
        
        

            
            # create a binary pickle file 
            f = open("results/file.pkl","wb")

            # write the python object (dict) to pickle file
            pickle.dump(RESULT,f)

            # close file
            f.close()
        
print('FINISH !!!')
