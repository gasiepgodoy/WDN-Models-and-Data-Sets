import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GRUCell(nn.Module):
    
    def __init__(self, input_size, hidden_size):    
        super(GRUCell, self).__init__()
        
        self.hidden_size = hidden_size
        
        # Layers                                 
        self.linear_z = nn.Linear(input_size+hidden_size, hidden_size)
        self.linear_r = nn.Linear(input_size+hidden_size, hidden_size)
        self.linear = nn.Linear(input_size+hidden_size, hidden_size)
        
        self._initialization()
        
    def _initialization(self):        
        a = -np.sqrt(1/self.hidden_size)
        b = np.sqrt(1/self.hidden_size)        
        torch.nn.init.uniform_(self.linear_z.weight, a, b)
        torch.nn.init.uniform_(self.linear_z.bias, a, b)        
        torch.nn.init.uniform_(self.linear_r.weight, a, b)
        torch.nn.init.uniform_(self.linear_r.bias, a, b)        
        torch.nn.init.uniform_(self.linear.weight, a, b)
        torch.nn.init.uniform_(self.linear.bias, a, b)                

    def forward(self, input_, hidden_state):  
        
            inputs_and_prev_state = torch.cat((input_, hidden_state), -1)
            
            # z = sigma(W_z * a + U_z * h(t-1)) (3)
            update_gate = self.linear_z(inputs_and_prev_state).sigmoid()            
            # r = sigma(W_r * a + U_r * h(t-1)) (4)
            reset_gate = self.linear_r(inputs_and_prev_state).sigmoid()            
            # h_hat(t) = tanh(W * a + U*(r o h(t-1))) (5) 
            new_hidden_state = self.linear(torch.cat((input_, reset_gate * hidden_state), -1)).tanh()           
            # h(t) = (1-z) o h(t-1) + z o h_hat(t) (6)
            output = (1 - update_gate) * hidden_state + update_gate * new_hidden_state               
            
            return output   
       
class GGNNModel(nn.Module):
           
    def __init__(self, attr_size, hidden_size, propag_steps):    
        super(GGNNModel, self).__init__()        
        
        self.attr_size = attr_size
        self.hidden_size = hidden_size
        self.propag_steps = propag_steps            
        
        # Layers
        self.linear_i = nn.Linear(attr_size,hidden_size)
        self.gru = GRUCell(2*hidden_size, hidden_size)        
        self.linear_o = nn.Linear(hidden_size, 1)
        
        self._initialization()
        
    def _initialization(self):       
        torch.nn.init.kaiming_normal_(self.linear_i.weight)
        torch.nn.init.constant_(self.linear_i.bias, 0)
        torch.nn.init.xavier_normal_(self.linear_o.weight)
        torch.nn.init.constant_(self.linear_o.bias, 0)          
    
    def forward(self, attr_matrix, adj_matrix):
        
        '''        
        attr_matrix of shape (batch, graph_size, attributes dimension)
        adj_matrix of shape (batch, graph_size, graph_size)   
        
            > Only 0 (nonexistent) or 1 (existent) edge types
        
        '''
        mask = (attr_matrix[:,:,0] != 0)*1
        
        A_in = adj_matrix.float() 
        A_out = torch.transpose(A_in,-2,-1) 
        
        if len(A_in.shape) < 3:
            A_in = torch.unsqueeze(A_in,0)  
            A_out = torch.unsqueeze(A_out,0)  
        if len(attr_matrix.shape) < 3:
            attr_matrix = torch.unsqueeze(attr_matrix,0)
               
        #print(np.shape(attr_matrix))
        hidden_state = self.linear_i(attr_matrix.float()).relu()
                
        #print(np.shape(self.linear_i(attr_matrix.float()).relu()))
        for step in range(self.propag_steps):            
            # a_v = A_v[h_1 ...  h_|V|]
            
            #print(np.shape(A_in))
            #print(np.shape(A_out))
            #exit()
            a_in = torch.bmm(A_in, hidden_state)
            a_out = torch.bmm(A_out, hidden_state)
       
            # GRU-like update
            hidden_state = self.gru(torch.cat((a_in, a_out), -1), hidden_state)
                    
        # Output model
        output = self.linear_o(hidden_state).squeeze(-1)  
        output = output + (mask + 1e-45).log() # Mask output
        output = output.log_softmax(1)        
        return output       
     
       
