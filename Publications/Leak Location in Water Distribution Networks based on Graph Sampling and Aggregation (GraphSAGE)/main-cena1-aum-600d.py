from sklearn.preprocessing import Normalizer
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from model import GGNN
from model import GraphSAGE
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
import time

start_time = time.time()


now = datetime.now()
d_print = now.strftime("%d%m_%H%M")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

folder='torch_data/'
Save_folder ='work_dir/'
# Renomear o nome correto no train dataset e val dataset
parser = argparse.ArgumentParser(description='linkage indetification')
parser.add_argument('--n_epochs', type=int, default=300, help='Number of Epochs')
parser.add_argument('--windows_size', type=int, default=5, help='Neighborhood Interation')
parser.add_argument('--hidden_layer_size', type=int, default=256, help='Number of Neurons in Hidden Layer')
parser.add_argument('--batch_size', type=int, default=300, help='Batch size')
parser.add_argument('--Train_dataset', type=str, default='seed10Cenas1-aum-600.pt', help='Dataset with linkage mensurements in .pt format')
parser.add_argument('--Train_adj', type=str, default='adj_matrices.pt', help='Adjacency Matrix of Train Datast')
parser.add_argument('--Val_dataset', type=str, default='seed10Cenas1-aum-600.pt', help='Dataset with linkage mensurements in .pt format')
parser.add_argument('--Val_adj', type=str, default='adj_matrices.pt', help='Adjacency Matrix of Test Datast')
parser.add_argument('--k', type=int, default=10, help='number of folds')

#parser.add_argument('--Test_dataset', type=str, default='seed575Dataset18_Noise.pt', help='Dataset with linkage mensurements in .pt format')
#parser.add_argument('--Test_adj', type=str, default='adj_matrices.pt', help='Adjacency Matrix of Test Datast')

args = parser.parse_args()

batch_size = args.batch_size
num_epochs = args.n_epochs


model = GGNN.GGNNModel(1, args.hidden_layer_size, args.windows_size).to(device)
#model = GraphSAGE.GraphSAGEModel(1, args.hidden_layer_size, 'lstm', args.windows_size).to(device)
print(model) 

TrainDataset = torch.load(folder + args.Train_dataset)
ValDataset = torch.load(folder + args.Val_dataset)
#TestDataset = torch.load(folder + args.Test_dataset)

#train_loader = torch.utils.data.DataLoader(TrainDataset, batch_size=batch_size, shuffle=True)
#val_loader = torch.utils.data.DataLoader(ValDataset, batch_size=batch_size, shuffle=True)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

adjacency = torch.load(folder + args.Train_adj)

################################ k-fold
dataset=ConcatDataset([TrainDataset, ValDataset])
#k=args.k                                                    # number of folds
splits=KFold(n_splits=args.k,shuffle=True,random_state=42)
foldperf={}                                                 # dictionary to save statistics
folderconf={}
################################

    
def train_epoch(model, device, trainLoader, loss_fn, optimizer, batch_size, adj_matrices): 
    #print('TRAINNING START')   
    
    model.train()
    
    train_pred = []
    train_correct = []
    #Train_acc_epoch = []
    for epoch in range(num_epochs):
        train_loss = train_acc = 0.0
        
        #for step in range(batch_size):
            # Get batch data
        X, label, idx = next(iter(trainLoader))

        A = adj_matrices[idx]

        Y = model(X.cuda(), A.cuda())                                 # Result is a weigthed vector given by softmax

        predictions = torch.exp(Y).argmax(-1).cpu()                              

        loss = criterion(Y, label.cuda())                             # Compute loss
        
        optimizer.zero_grad()                                         # Clear the gradients

        loss.backward()                                               # Calcule gradients
        optimizer.step()                                              # Update Weights

        train_loss += loss.item()                                     # Calculate Loss

        train_acc += ((predictions == label).sum())/len(label)    # Accuracy
        #train_acc += ((predictions == label).sum())    # Accuracy
        
        train_pred.extend(predictions)
        train_correct.extend(label)
        
        #print('train_Acc', train_acc)
            
        
        return train_loss, train_acc, train_pred, train_correct
        
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
    

if __name__ == '__main__':
    
    

    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(dataset)))):
        
        #print('fold {}'.format(fold+1))
                
        train_sample = SubsetRandomSampler(train_idx)
        val_sample = SubsetRandomSampler(val_idx)
        
        train_loader = DataLoader(dataset,batch_size=batch_size, sampler=train_sample)
        val_loader = DataLoader(dataset,batch_size=batch_size, sampler=val_sample)   
           
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        
        history = {'train_loss': [], 'val_loss': [],'train_acc':[],'val_acc':[]}
        
        data_conf = {'train_pred': [], 'train_corr': [], 'val_pred': [], 'val_corr':[]}
        
        
        for epoch in range(num_epochs):
            train_loss, train_acc, train_pred, train_label = train_epoch(model, device, train_loader, criterion, optimizer, batch_size, adjacency)
            val_loss, val_acc, val_pred, val_label = validation_epoch(model, device, val_loader,criterion, adjacency)
            
            
            train_loss = train_loss
            val_loss = val_loss
            train_acc = train_acc*100
            val_acc = val_acc*100
                       

            
            print("Fold:{} Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f} AVG Training Acc {:.2f} % AVG Test Acc {:.2f} %".format(fold+1,epoch+1, num_epochs, 
                                                                    round(train_loss,3),
                                                                    round(val_loss,3),
                                                                    train_acc,
                                                                    val_acc))

            
    
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            data_conf['train_pred'].append(train_pred)
            data_conf['train_corr'].append(train_label)
            data_conf['val_pred'].append(val_pred)
            data_conf['val_corr'].append(val_label)

        
        foldperf['fold{}'.format(fold+1)] = history
        folderconf['fold{}'.format(fold+1)] = data_conf
    
    # calculate the average score in every fold & average score of all folds
    
    sc_tl, sc_ta, sc_vl, sc_va = [], [], [], []
    
    nf=args.k # number of folds
    
    torch.save(data_conf, 'work_dir/data_to_Mconf_aum_600_GraphSAGE_Lstm.pt')  # renomear
    
    for f in range(1,nf+1):
        sc_tl.append(np.mean(foldperf['fold{}'.format(f)]['train_loss']))
        sc_ta.append(np.mean(foldperf['fold{}'.format(f)]['val_loss']))
        
        sc_vl.append(np.mean(foldperf['fold{}'.format(f)]['train_acc']))
        sc_va.append(np.mean(foldperf['fold{}'.format(f)]['val_acc']))
    
    
    print('\n')
    print('Performance of {} fold cross validation'.format(args.k))
    print('AVG Training Loss: {:.3f} \t AVG Val Loss: {:.3f} \t AVG Training Acc: {:.2f} \t AVG Val Acc: {:.2f}'.format(np.mean(sc_tl), np.mean(sc_ta), np.mean(sc_vl), np.mean(sc_va)))
    #print('history', history)
    
    diz_ep = {'train_loss_ep':[],'test_loss_ep':[],'train_acc_ep':[],'test_acc_ep':[]}
    
    for i in range(num_epochs):
        diz_ep['train_loss_ep'].append(np.mean([foldperf['fold{}'.format(f+1)]['train_loss'][i] for f in range(args.k)]))
        diz_ep['test_loss_ep'].append(np.mean([foldperf['fold{}'.format(f+1)]['val_loss'][i] for f in range(args.k)]))
        diz_ep['train_acc_ep'].append(np.mean([foldperf['fold{}'.format(f+1)]['train_acc'][i] for f in range(args.k)]))
        diz_ep['test_acc_ep'].append(np.mean([foldperf['fold{}'.format(f+1)]['val_acc'][i] for f in range(args.k)]))    
    
    torch.save(model, folder + 'Cena_1_Aum_600_model_state_dict.pt')   # renomear
    torch.save(diz_ep, folder + 'Cena_1_Aum_600_training_statistics.pt') # renomear
    
    # Plot losses
    plt.figure(figsize=(10,8))
    plt.plot(diz_ep['train_loss_ep'], label='Treinamento')
    plt.plot(diz_ep['test_loss_ep'], label='Teste')
    #plt.semilogy(diz_ep['train_loss_ep'], label='Train')
    #plt.semilogy(diz_ep['test_loss_ep'], label='Test')
    plt.xlabel('Epocas')
    plt.ylabel('Perda')
    #plt.xlim(0, 300)
    #plt.ylim(0, 3)
    #plt.grid()
    plt.legend()
    #plt.title('GGNN cross validation  loss')
    plt.show() 
    #plt.savefig('work_dir/' + d_print +'loss.png')
    
    # Plot accuracies
    plt.figure(figsize=(10, 8))
    plt.plot(diz_ep['train_acc_ep'], label='Treinamento')
    plt.plot(diz_ep['test_acc_ep'], label='Teste')
    #plt.semilogy(diz_ep['train_acc_ep'], label='Train')
    #plt.semilogy(diz_ep['test_acc_ep'], label='Val')
    plt.xlabel('Epocas')
    plt.ylabel('Acurácia')
    #plt.xlim(0, 300)
    #plt.ylim(0, 100)
    #plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.2f}'.format(x)))
    #plt.grid()
    plt.legend()
    #plt.title('GGNN cross validation accuracy')
    plt.show()
    #plt.savefig('work_dir/' + d_print +'Acc.png')

    
    #m_name = 'work_dir/' + d_print + '_k_cross_GGNN.pt'
    #torch.save(model,n_name)

print("Finish")

nome_arquivo = r"C:\Users\GASI\Documents\Weliton Rodrigues\Projeto\Projeto de Pesquisa\Dataset CBA 2022\Pesos\modelo_GraphSAGE_Lstm.pth"
torch.save(model.state_dict(), nome_arquivo)

# Termina a contagem de tempo e calcula o tempo total decorrido
end_time = time.time()
elapsed_time = end_time - start_time
print("Tempo total de treinamento: {:.2f} segundos".format(elapsed_time))