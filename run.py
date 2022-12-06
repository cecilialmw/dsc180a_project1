import sys
import os
import json

sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
import cv2
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix

import torch 
from  torchvision import transforms, models
import torch.nn as nn
from torch.optim import Adam
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader



from create_dataloader import ImportData
from create_dataloader import create_loader

from build_model import resnet152
from build_model import vgg19
from build_model import vgg16
from build_model import train_val_model

from evaluate_test import plot_both_loss
from evaluate_test import test_model
from evaluate_test import test_mae
from evaluate_test import plot_pearson_r

TEST_PATH = '/test/testdata/test.csv'
TRAIN_PATH = '/test/testdata/train.csv'
VAL_PATH = '/test/testdata/val.csv'

def main(targets):
    '''
    Runs the main project pipeline logic, given the targets.
    targets must contain: 'data', 'analysis', 'model'. 
    
    `main` runs the targets in order of data=>analysis=>model.
    '''

    print(targets)
    model_name = targets[1]
    print('model:' + model_name)
    if model_name == 'vgg16':
        model_test = vgg16()
    elif model_name == 'vgg19':
        model_test = vgg19()
    else:
        model_test = resnet152()

    resolution = targets[2]
    print('res: ' + resolution)
    resolution = int(resolution)
    if rosolution == 64:
        res_name = '64x64'
    else:
        res_name = '224x224'
    
    lr = 8e-5
    if model_name == 'resnet152':
        if resolution == 256:
            lr = 8e-4
    
    
    if 'test' in targets:
        train_loader, val_loader, test_loader = create_loader(TRAIN_PATH, VAL_PATH, TEST_PATH, resolution, 8)
        
        model_trained, train_loss, val_loss = train_val_model(model = model_test,
                                                              batch_size = 8,
                                                              num_epochs = 20,  
                                                              learning_rate = lr,
                                                              train_loader = train_loader, val_loader = val_loader)
        
        plot_both_loss(train_loss_resnet152_224, val_loss_resnet152_224, model_name, res_name)
        y_test, y_true, test_mae_out, pearson, roc, f1 = test_analysis(model_trained, model_name, resolution, test_loader, threshold = 400)
        
        test_mae_out = test_mae(y_test, y_true)
        
        
        with open('output/results.txt', 'w') as f:
            f.write('Model -- ' + model_name + ', Resolution -- ' + res_name)
            f.write('The training loss are: ' + str(train_loss))
            f.write('The validation loss are: ' + str(val_loss))
            f.write('The true log BNPP value are: ' + str(y_true))
            f.write('The inferred log BNPP value are: ' + str(y_test))
            f.write('The test MAE is: ' + str(test_mae_out))
            f.write('The pearson r coefficient is: ' + str(pearson))
            f.write('The ROC-AUC score is: ' + str(roc))
            f.write('The F1 score is: ' + str(f1))
        
    return 

        


if __name__ == '__main__':
    # run via:
    # python main.py data features model
    targets = sys.argv[1:]
    main(targets)