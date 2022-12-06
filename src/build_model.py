import sys

sys.path.insert(0, 'src')

import numpy as np
from tqdm.notebook import tqdm

import torch 
from  torchvision import transforms, models
import torch.nn as nn
from torch.optim import Adam
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class resnet152(nn.Module):

    """

    Best to use pre-trained

    """

    def __init__(self):

        super().__init__()

        self.model = models.resnet152(pretrained=True)

        # initialize new output layer

        #self.model.classifier[6] = nn.Linear(2048, 1)
        layers = np.array([layer for layer in self.model.children()])

        for layer in layers[:-4]:

            for param in layer.parameters():
                
                # Change parameters for all layers
                param.requires_grad = False
        
        for layer in layers[-4][:-4]:
            for param in layer.parameters():
                param.requires_grad = False
                
        self.model.fc = nn.Linear(2048, 1)
        
        #num_open_param = 0
        
#         for layer in layers[-3:]:
#             for param in layer.parameters():
#                 num_open_param += 1
#         print('Num Open Parameters: ', num_open_param)


    def forward(self, x):

        x = self.model(x)

        return x
    
    
### Modifying model
class vgg19(nn.Module):

    """

    Best to use pre-trained

    """

    def __init__(self):

        super().__init__()

        self.model = models.vgg19(pretrained=True)

        # initialize new output layer

        self.model.classifier[6] = nn.Linear(4096, 1)


        for layer in self.model.children():

            for param in layer.parameters():
                
                # Change parameters for all layers
                param.requires_grad = True


    def forward(self, x):

        x = self.model(x)

        return x

    
    
### Modifying model
class vgg16(nn.Module):

    """

    Best to use pre-trained

    """

    def __init__(self):

        super().__init__()

        self.model = models.vgg16(pretrained=True)

        # initialize new output layer

        self.model.classifier[6] = nn.Linear(4096, 1)


        for layer in self.model.children():

            for param in layer.parameters():
                
                # Change parameters for all layers
                param.requires_grad = True


    def forward(self, x):

        x = self.model(x)

        return x
    
    
    
    
    
    
def train_val_model(model, batch_size, num_epochs, learning_rate, 
                    train_loader, val_loader):

    model.to(device)
    all_train_loss = []
    all_val_loss = []

    optimizer = Adam(model.parameters(), lr = learning_rate)
#     scaler = torch.cuda.amp.GradScaler()
    for epoch in tqdm(range(num_epochs)):

        ## training set
        total_train_loss = 0
        batch_num = 0
        model.train()
        for i, (data, labels) in enumerate(train_loader):
            batch_num += 1
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()
#                 data = data.to('cuda:0', non_blocking=True)
#                 labels = target.to('cuda:0', non_blocking=True)

            optimizer.zero_grad()
#             with torch.cuda.amp.autocast():
#                 target = model(data)
#                 loss = torch.abs(torch.tensor(1 + labels) - (1 + target)).mean()
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()

            target = model(data)
            
#             if i == 0:
#                 print(target.detach().cpu())
#                 print(torch.abs(torch.tensor(labels) - target))
#                 print(torch.abs(torch.tensor(labels) - target).mean())
            #loss = torch.abs(torch.tensor(1 + labels) - (1 + target)).mean()
            loss = torch.abs(torch.tensor(labels) - target).mean()

            loss.backward() # backward() calculates the derivative for that single value and adds it to the previous one.
            optimizer.step()

            total_train_loss += float(loss) # accumulate the total loss for this epoch.

        if epoch == 0:
            print("Total # of training batch: ", i + 1)

        all_train_loss.append(total_train_loss / batch_num)

        
        ## validation set
        batch_num = 0
        total_val_loss = 0
        model.eval()
        for i, (data, labels) in enumerate(val_loader):
            batch_num += 1
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()

            target = model(data)

            loss = torch.abs(torch.tensor(labels) - target).mean()

            total_val_loss += float(loss) # accumulate the total loss for this epoch.

        if epoch == 0:
            print("Total # of validation batch: ", i + 1)

        all_val_loss.append(total_val_loss / batch_num)
        
    
    #plot_both_loss(all_train_loss, all_val_loss)
        
    return model, all_train_loss, all_val_loss