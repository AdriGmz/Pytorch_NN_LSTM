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
path_data    = path_base + '../data/'
path_images  = path_base + '../images/2_TimeSeries/'
path_results = path_base + '../results/2_TimeSeries/'
