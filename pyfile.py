import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression as LR
import torch
from torch import nn
import torch.utils.data as data
import torch.nn.functional as F
from torch.nn import Module
import torch.optim as optim
import time

from torch.utils.data import Dataset

from sklearn.metrics import classification_report, roc_auc_score, f1_score, recall_score, precision_score, accuracy_score

from model import *
from data_processor import solve_sample
from dataset import *
from utlits import *


# def sample(num):
#     u = np.random.rand()
#
#     return 1 if u < num else 0
#
# def repeat(times = 1, possiblity=0.5):
#
#     temp = []
#
#     def sample(num):
#         u = np.random.rand()
#
#         return 1 if u < num else 0
#
#     for i in range(times):
#         a = sample(possiblity)
#         temp.append(a)
#
#     return temp
#
# def build(orginal_x, probilities, times=1, save=False):
#     treatments = [repeat(times, i) for i in probilities]
#     data = np.hstack((x.to_numpy(),treatments))
#     new = pd.DataFrame(data, columns=list(x.columns)
#                          + ["p"+str(i) for i in range(0,times)])
#
#     if save:
#         new.to_csv("new X.csv")
#         print("Successful saved!")
#
#     return new


class CustomLoss(Module):
    def __init__(self):
        super().__init__()
    def forward(self, predict_y, y,p):

        temp = y - torch.mul(predict_y, p)

        return torch.mean(torch.square(temp))


'''----------------------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------------------------'''

# Stage 1 to Stage 2
sampleData, prob = solve_sample()
stage2_data = sampleData[["workclass", "marital_status", "occupation", "relationship", "gender", "native_country", "age",
                  "education"]]

# Hyperparameter 
batch_size = 64
learning_rate = 0.005


#Prepare dataloader
train_data = stage2_data.to_numpy()
train_outcome = sampleData[">=50K"].to_numpy()

print(f'Number of training examples: {len(train_data)}')

train_dataset = imcomedataset(train_data, train_outcome,prob)

train_iterator = data.DataLoader(train_dataset,
                                 shuffle=True,
                                 batch_size=batch_size)

# Model
input_dim = stage2_data.shape[1]
output_dim = 1
model = MLP(input_dim, output_dim)

print(f'The model has {count_parameters(model):,} trainable parameters')


optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = CustomLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)
criterion = criterion.to(device)

'''----------------------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------------------------'''




def train(model, iterator, optimizer, criterion, device):

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for i,(x, y,p) in enumerate(iterator):

        x = x.to(device)
        y = y.to(device)
        p = p.to(device)


        y = torch.unsqueeze(y, 1)
        p = torch.unsqueeze(p, 1)

        optimizer.zero_grad()

        y_pred, _ = model(x)

        loss = criterion(y_pred, y, p)
        #loss = criterion(y_pred, y)

        acc = calculate_accuracy(y_pred, y)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion, device):

    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():

        for i,(x, y,p) in enumerate(iterator):

            x = x.to(device)
            y = y.to(device)
            p = p.to(device)

            y = torch.unsqueeze(y, 1)
            p = torch.unsqueeze(p, 1)

            y_pred, _ = model(x)

            loss = criterion(y_pred, y)

            acc = calculate_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

'''----------------------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------------------------'''

EPOCHS = 100

best_valid_loss = float('inf')

for epoch in range(EPOCHS):

    start_time = time.monotonic()

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device)

    end_time = time.monotonic()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
