import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import os
import torch as tch
from Functions import *
import matplotlib
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor

#Gráficas
plt.close('all')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 11

#PATHS
path_base    = os.path.dirname(os.path.abspath(__file__)) + '/'
path_data    = path_base + 'data/'
path_images  = path_base + 'images/1_Images/'
path_results = path_base + 'results/1_Images/'

#Data
batch_size = 64

training_data = datasets.FashionMNIST( root=path_data, train=True,
    download=True, transform=ToTensor() )

testing_data = datasets.FashionMNIST(root=path_data, train=False,
    download=True, transform=ToTensor() )

train_data = DataLoader(training_data, batch_size=batch_size)
test_data = DataLoader(testing_data, batch_size=batch_size)

#====================================================================
# CONFIGURATION
#====================================================================
num_epochs = 2 
learning_rate = 0.001 #0.001 lr

#====================================================================
# Number of nodes for input and ouput layers
#====================================================================
n_input = 1; 
flat_data = False # False for time-series, True for 3D data
if   len(training_data.data.shape) == 3:
    n_input = training_data.data.shape[-1]*training_data.data.shape[-2]
    flat_data = True
elif len(training_data.data.shape) == 2: 
    n_input = training_data.data.shape[-1] # number of features

n_output  = len(training_data.classes) # number of classes

#====================================================================
# Example of number of hidden layers and number of nodes
# Dimention of n_nodes = N_layer - 1
# Number of nodes depends of your data
#====================================================================
# N_layer   = 4; n_nodes  = [50, 100, 70]
# N_layer   = 3; n_nodes  = [10, 40]
# N_layer   = 2; n_nodes  = [15]
# N_layer   = 1; n_nodes  = [] 
#====================================================================
N_layer  = 2 # Number of layers
#n_nodes to test for each layer
# nod_lay1 = [0];   nod_lay2 = [0];    nod_lay3 = [0] #Example for N_layer = 1
nod_lay1 = range(40,100,10);   nod_lay2 = [0];    nod_lay3 = [0]  #Example for N_layer = 2
# nod_lay1 = range(20,100,5); nod_lay2 = range(20,100,5); nod_lay3 = [0] #Example for N_layer = 3
res_indx = [f'{N_layer}L_{n1}_{n2}_{n3}' for n1 in nod_lay1 for n2 in nod_lay2 for n3 in nod_lay3]

#DataFrame to save the test result of each configuration, in order to analyze them later
DF_results = pd.DataFrame(index=res_indx, columns = ['NN_ac', 'NN_corr', 'LSTM_ac', 'LSTM_corr'])

#====================================================================
# Setting models for each configuration 
#====================================================================
print_res  = False # Show test_loss, test_accuracy and test_correlation
get_figure = False # Save figure with the results of the last test steps
for n_nodes_1 in nod_lay1:
    for n_nodes_2 in nod_lay2:
        for n_nodes_3 in nod_lay3:
            n_nodes = [n_nodes_1, n_nodes_2, n_nodes_3]
            title = f'{N_layer}L_{n_nodes_1}_{n_nodes_2}_{n_nodes_3}'
            nodes = '_'.join(list(map(str, n_nodes)))
            print ('-------------------------------------------------------------')
            print (str(N_layer)+' layers', nodes + ' nodes')
            print ('-------------------------------------------------------------')
            
            print ('NN')
            model1 = MyNN(n_input, n_output, N_layer, n_nodes, flat=flat_data)
            # print (model1)
            criterion1 = nn.CrossEntropyLoss() # Classification
            optimizer1 = tch.optim.Adam(model1.parameters(), lr=learning_rate) 

            for epoch in range(num_epochs):
                for batch_NN, (x_NN, y_NN) in enumerate(train_data):
                    outputs1 = model1(x_NN) #forward pass
                    loss1 = criterion1(outputs1, y_NN)
                    # Backpropagation
                    optimizer1.zero_grad() #para que no tome dos veces el gradiente
                    loss1.backward() # propaga el error a cada parámetro (gradientes)
                    optimizer1.step() # updates - reajusta los parámetros en función del gradiente

            print(f"Train loss: {loss1.item():>5f}")
            test_loss1, correct1, test_corr1, serie_pred1, pred_probab1 = test_loop(model1, criterion1, testing_data, res_print = print_res)

            print ('\n-------------------------------------------------------------\n')
            print ('LSTM')
            N_lay_lstm = 1
            n_lstm  = 5
            model2 = MyLSTM(n_input, n_output, N_layer, n_nodes, N_lay_lstm, n_lstm, flat=flat_data)
            # print (model2)
            criterion2 = nn.CrossEntropyLoss() # Classification
            optimizer2 = tch.optim.Adam(model2.parameters(), lr=learning_rate) 

            for epoch in range(num_epochs):
                for batch, (xi, yi) in enumerate(train_data):
                    xi, yi = check_lstm_dim(xi, yi)
                    outputs2 = model2(xi) #forward pass
                    loss2 = criterion2(outputs2, yi.reshape((len(yi))))
                    # Backpropagation
                    optimizer2.zero_grad() #para que no tome dos veces el gradiente
                    loss2.backward() # propaga el error a cada parámetro (gradientes)
                    optimizer2.step() # updates - reajusta los parámetros en función del gradiente

            print(f"Train loss: {loss2.item():>5f}")
            test_loss2, correct2, test_corr2, serie_pred2, pred_probab2 = test_loop(model2, criterion2, testing_data,  mod_lstm = True, flat = True, res_print = print_res)

            print ('\n-------------------------------------------------------------\n')

            #Figure of the last testing batch
            if get_figure:
                plt.figure(figsize=(8,4))
                plt.title(f'{title}\n NN - AC: {(100*correct1):>0.1f}% Corr: {test_corr1:>0.3f}\nLSTM - AC: {(100*correct2):>0.1f}% Corr: {test_corr2:>0.3f}')
                plt.plot(serie_pred1[-batch_size:], 'r', label = 'NN')
                plt.plot(serie_pred2[-batch_size:], 'b', label = 'LSTM')
                plt.plot(testing_data.targets[-batch_size:], 'k', label = 'targets')
                plt.legend()
                plt.savefig(path_images + title+'.png',bbox_inches='tight', dpi=100)
                plt.close('all')

            #Add accuracy and correlation to DF for each model 
            DF_results.loc[title]['NN_ac'] = np.round(correct1*100, 2)
            DF_results.loc[title]['LSTM_ac'] = np.round(correct2*100, 2)
            DF_results.loc[title]['NN_corr'] = np.round(test_corr1,3)
            DF_results.loc[title]['LSTM_corr'] = np.round(test_corr2,3)

DF_results.to_csv(path_results+'Results_NN_LSTM_'+str(N_layer)+'L.csv')

best = get_best_models (DF_results)
print (best)
