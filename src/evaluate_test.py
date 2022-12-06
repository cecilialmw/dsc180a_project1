import sys
import os
import json

sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy.stats import pearsonr
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

import torch 
from  torchvision import transforms, models
import torch.nn as nn
from torch.optim import Adam
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def plot_both_loss(all_train_loss, all_val_loss, model_name, resolution):
    plt.figure(figsize=(20, 10))
    sns.set(style="whitegrid")
    epoch_num = len(all_train_loss)
    df = pd.DataFrame({'x':range(epoch_num),
                    'train_loss':all_train_loss,
                      'val_loss':all_val_loss})
    df = df.set_index('x')
    
    train_val_loss = sns.lineplot(data=df, linewidth=2.5)

    ## now label the y- and x-axes.
    plt.ylabel('Customize MAE Loss')
    plt.xlabel('Epoch Number')
    plt.title('MAE Loss of {} with resolution {}'.format(model_name, resolution))
    plt.savefig('output/loss_{}_{}.png'.format(model_name, resolution))
    plt.show()


    return
    
    
def test_model(model, loader):
    n = 0
    y_test = []
    y_true = []
    y_image = []
    all_test_loss = []
    model.eval()
    for i, (data, labels) in enumerate(loader):
        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()

        target = model(data)

        y_test.append(target[0].detach().cpu())
        y_true.append(labels[0].detach().cpu())
        # if i < 5:
        #     y_image.append(data.detach().cpu())
                
    return np.array(y_test), np.array(y_true), y_image

def plot_BNPP(y_test, y_true, y_image, model_name, resolution):    
    BNPP_true = np.around(np.power(10, y_true[:5]), 0)
    BNPP_test = np.around(np.power(10, y_test[:5]), 0)
    
    fig_size = (10, 10)
    fig, axs = plt.subplots(1, 2, figsize = fig_size)

    axs[0].imshow(y_image[0][0][0], cmap='gray')
    axs[0].title.set_text('True = {}, Inferred = {}'.format(BNPP_true[0], BNPP_test[0]))
    axs[1].imshow(y_image[1][0][0], cmap='gray')
    axs[1].title.set_text('True = {}, Inferred = {}'.format(BNPP_true[1], BNPP_test[1]))
    
    fig.suptitle('Model {} with resolution {}'.format(model_name, resolution), y=0.73)
    plt.savefig('output/BNPP_{}_{}.png'.format(model_name, resolution))
    plt.show()

    
    return


def test_mae(y_test, y_true):
    test_mae = np.abs(y_test - y_true).mean()
    
    return test_mae



def plot_pearson_r(y_test, y_true, model_name, resolution, color = "#4CB391"):
    corr, _ = pearsonr(y_true, y_test)
    corr = np.around(corr, 3)
    
    sns.scatterplot(x=y_true, y=y_test, color=color)
    plt.title('model {} with resoltuon {} have r = {}'.format(model_name, resolution, corr))
    plt.xlabel('True BNPP')
    plt.ylabel('Inferred BNPP')
    plt.savefig('output/pearson_{}_{}.png'.format(model_name, resolution))
    plt.show()

    return corr

def plot_roc_curve(y_test, y_true, threshold, model_name, resolution):

    y_true_bi = np.where(y_true <= np.log10(threshold), 0, 1)

    fpr, tpr, threshold = roc_curve(y_true_bi, y_test, drop_intermediate = False)
    roc_auc = roc_auc_score(y_true_bi, y_test)

    plt.figure(1)
    plt.plot([0, 1], [0, 1])
    plt.plot(fpr, tpr, label='{}(area = {:.3f})'.format(model_name, roc_auc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve of model {} with resolution {}'.format(model_name, resolution))
    plt.legend(loc='best')
    plt.savefig('output/roc_{}_{}.png'.format(model_name, resolution))

    plt.show()

    return roc_auc

def plot_confusion_matrix(y_test, y_true, threshold, model_name, resolution):
    y_true_bi = np.where(y_true <= np.log10(threshold), 0, 1)
    y_test_bi = np.where(y_test <= np.log10(threshold), 0, 1)
    
    cm = confusion_matrix(y_true_bi, y_test_bi)
    tn, fp, fn, tp = confusion_matrix(y_true_bi, y_test_bi).ravel()
    f1_score = tn / (tn + 0.5 * (fp + fn))
    f1_score = np.around(f1_score, 3)

    sns.heatmap(cm, annot=True, cmap = 'Blues', fmt="d")
    plt.title('Confusion matrix of model {} with resolution {}'.format(model_name, resolution))
    plt.xlabel('Prediction')
    plt.ylabel('True')
    plt.savefig('output/cm_{}_{}.png'.format(model_name, resolution))

    plt.show()

    return f1_score

def test_analysis(model, model_name, resolution, loader, threshold = 400):
    # get test results
    y_test, y_true, y_image = test_model(model, loader)
    print(y_test)
    print(y_true)
    
    # get test MAE
    #test_mae = np.around(np.abs(y_test - y_true).mean(), 3)
    #print('Test MAE of model {} is {}'.format(model_name, test_mae))
    
    # plot image with true bnpp and inferred bnpp
    #plot_BNPP(y_test, y_true, y_image, model_name, resolution)
    
    # plot pearson r
    #pearson = plot_pearson_r(y_test, y_true, model_name, resolution, color = "#4CB391")
    pearson = 0
    
    # plot auc-roc
    roc = plot_roc_curve(y_test, y_true, threshold, model_name, resolution)
    
    # plot confusion matrix
    f1 = plot_confusion_matrix(y_test, y_true, threshold, model_name, resolution)

    return y_test, y_true, pearson, roc, f1

