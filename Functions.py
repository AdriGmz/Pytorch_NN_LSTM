
import numpy as np
import pandas as pd
import os
import torch as tch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def check_lstm_dim (data_x, target):
    if data_x.shape[1] != 1:
        new_data_shape = tuple(np.insert(data_x.shape, 1,1))
        data_x =  tch.reshape(data_x, new_data_shape)
    if target.shape[-1] != 1:
        new_target_shape = tuple(np.insert(target.shape, 1,1))
        target = tch.reshape(target, new_target_shape)
    return data_x, target

def get_best_models (df_results):
    columns = df_results.keys()
    index   = df_results.index
    serie   = pd.Series(index = columns)
    for cc in columns:
        data = [index[indx] for indx, item in enumerate(df_results[[cc]]) if item == max(df_results[[cc]])]
        serie.loc[cc] = data[0]
    return serie
    

def test_loop(model, criterion, data_test, mod_lstm = False, flat = False, res_print = True):
    tch.no_grad()
    #Check dimentions for lstm
    if mod_lstm:
        data_x, target =  check_lstm_dim(data_test.data, data_test.targets)  
    else:
        data_x, target = data_test.data, data_test.targets 
    #Check dimentions for targets
    target = target.reshape((len(target))) if flat else target
    #Run the model
    outputs = model(data_x.float()) 
    pred_probab = nn.Softmax(dim = 1)(outputs) # transforma pred a probabilidades
    serie_pred  = pred_probab.argmax(1)
    test_loss   = criterion(outputs, target).item() # medida del error
    correct     = (serie_pred == target).type(tch.float).sum().item()
    correct    /= len(target)
    test_corr   = np.corrcoef(serie_pred, target)[0,1]
    if res_print:
        print(f"Test loss: {test_loss:>5f}")
        print(f"Test Accuracy: {(100*correct):>0.1f}%")
        print ("Test Correlation: ", test_corr)
    return test_loss, correct, test_corr, serie_pred, pred_probab

class MyStack():
    def __init__(self, n_input, n_output, N_layer, n_linear):
        self.model = ''
        self.N_layer = N_layer
        self.n_linear = n_linear
        if N_layer == 1:
            self.sequence  = nn.Sequential(nn.Linear(n_input, n_output))
        elif N_layer == 2:
            self.sequence  = nn.Sequential(nn.Linear(n_input, n_linear[0]),
                            nn.ReLU(), nn.Linear(n_linear[0], n_output))
        elif N_layer == 3:
            self.sequence  = nn.Sequential(nn.Linear(n_input, n_linear[0]),
                            nn.ReLU(), nn.Linear(n_linear[0], n_linear[1]),
                            nn.ReLU(), nn.Linear(n_linear[1], n_output))
        elif N_layer == 4:
            self.sequence  = nn.Sequential(nn.Linear(n_input, n_linear[0]),
                            nn.ReLU(), nn.Linear(n_linear[0], n_linear[1]),
                            nn.ReLU(), nn.Linear(n_linear[1], n_linear[2]),
                            nn.ReLU(), nn.Linear(n_linear[2], n_output))
        else:
            raise ValueError ('Función programada hasta para 4 capas')
        

class MyNN(nn.Module):
    """
    N_layer capas lineales intercaladas con N_layer - 1 capas ReLU
    """
    def __init__(self, n_input, n_output, N_layer, n_linear, flat = False):
        super(MyNN, self).__init__()
        self.flat = flat
        self.flatten  = nn.Flatten()
        self.n_output = n_output
        # Capas de entrada y de salida
        Stack = MyStack(n_input, n_output, N_layer, n_linear)
        self.stack = Stack.sequence

    def forward(self, x):
        x = self.flatten(x) if self.flat else x
        result = self.stack(x)
        result = self.flatten(result) if self.n_output == 1 else result
        return result

class MyLSTM(nn.Module):
    def __init__(self, n_input, n_output, N_layer, n_linear, N_lay_lstm, n_lstm, flat = False):
        super(MyLSTM, self).__init__()
        self.flat = flat
        self.flatten  = nn.Flatten()
        self.N_lay_lstm = N_lay_lstm #number of layers
        self.n_lstm = n_lstm #hidden nodes state

        self.lstm = nn.LSTM(n_input, n_lstm, N_lay_lstm, batch_first=True) #lstm
        # Capas de entrada y de salida
        Stack = MyStack(n_lstm, n_output, N_layer, n_linear)
        self.stack = Stack.sequence
    
    def forward(self,x):
        h_0 = tch.zeros(self.N_lay_lstm, x.size(0), self.n_lstm) #hidden state
        c_0 = tch.zeros(self.N_lay_lstm, x.size(0), self.n_lstm) #internal state
        # Propagate input through LSTM
        x = self.flatten(x) if self.flat else x
        if len(x.shape) == 2:
            x = tch.reshape(x, tuple(np.insert(x.shape, 1,1)))
        output, _ = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        output = output.view(-1, self.n_lstm) #reshaping the data for Dense layer next
        out = self.stack(output)
        return out

class MyCategories(Dataset):
    def __init__(self, X_train, Y_train):
        self.data_x = tch.tensor(X_train, dtype = tch.float32)
        self.target = tch.tensor(Y_train)

    def set_time_shape (self):
        self.data_x = tch.reshape(self.data_x, (self.data_x.shape[0], 1,self.data_x.shape[1]))
        self.target = tch.reshape(self.target, (self.target.shape[0], 1))

    def __len__(self):
        """
        returns the number of samples in our dataset.
        """
        return len(self.target)

    def __getitem__(self, idx):
        """
        loads and returns a sample from the dataset at the given index idx
        """
        return self.data_x[idx], self.target[idx]


