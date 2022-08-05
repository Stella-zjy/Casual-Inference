import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression as LR
import torch
from torch import nn
import torch.utils.data as data
import torch.nn.functional as F

import torch.optim as optim
import time

from torch.utils.data import Dataset
from data_processor import solve_sample

from sklearn.metrics import classification_report, roc_auc_score, f1_score, recall_score, precision_score, accuracy_score

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

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        self.input_fc = nn.Linear(input_dim, 10)
        self.hidden_fc = nn.Linear(10, 10)
        self.output_fc = nn.Linear(10, output_dim)

    def forward(self, x):

        # x = [batch size, height, width]

        batch_size = x.shape[0]

        x = x.view(batch_size, -1)

        # x = [batch size, height * width]

        h_1 = F.relu(self.input_fc(x))
        # h_1 = [batch size, 250]

        h_2 = F.relu(self.hidden_fc(h_1))
        # h_2 = [batch size, 100]

        h3 = self.output_fc(h_2)
        # y_pred = [batch size, output dim]


        y_pred = F.sigmoid(h3)
        #y_pred = F.softmax(h3, dim=1)
        return y_pred, h3

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

from torch.nn import Module
class CustomLoss(Module):
    def __init__(self):
        super().__init__()
    def forward(self, predict_y, y,p):

        temp = y - torch.mul(predict_y, p)

        return torch.mean(torch.square(temp))


class imcomedataset(Dataset):
 
  def __init__(self,train_data,train_outcome):

    self.x_train = torch.tensor(train_data, dtype=torch.float32)
    self.y_train = torch.tensor(train_outcome, dtype=torch.float32)
    self.prob = torch.from_numpy(prob.to_numpy())
 
  def __len__(self):
    return len(self.y_train)
   
  def __getitem__(self,idx):
    return self.x_train[idx],self.y_train[idx],self.prob[idx]

'''----------------------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------------------------'''

# df = pd.read_csv("../Casual-Inference/data/income_data/modified_train.csv")
#
#
# y = df["education"]
# #x = df[["race","workclass", "fnlwgt", "marital_status", "occupation", "relationship", "age"]]
# x = df[["race","workclass",
#         "marital_status", "occupation", "relationship", "age"]]
#
# model = LR().fit(x, y)
#
# probilities = model.predict_proba(x)[:,1]
#
# treatments = [repeat(10, i) for i in probilities]
#
# new_x = build(x, probilities, 20)


# VALID_RATIO = 0.9
#
#
# n_train_examples = int(len(x) * VALID_RATIO)
# n_valid_examples = len(x) - n_train_examples

sampleData, prob = solve_sample()
stage2_data = sampleData[["workclass", "marital_status", "occupation", "relationship", "gender", "native_country", "age",
                  "education"]]

#train_data = new_x.to_numpy()
train_data = stage2_data.to_numpy()
train_outcome = sampleData[">=50K"].to_numpy()


print(f'Number of training examples: {len(train_data)}')


train_dataset = imcomedataset(train_data, train_outcome)

batch_size = 64

train_iterator = data.DataLoader(train_dataset,
                                 shuffle=True,
                                 batch_size=batch_size)

input_dim = stage2_data.shape[1]
#input_dim = x.shape[1] + 1
output_dim = 1


model = MLP(input_dim, output_dim)

print(f'The model has {count_parameters(model):,} trainable parameters')

optimizer = optim.Adam(model.parameters(), lr=0.005)


criterion = CustomLoss()
#criterion = nn.BCELoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = criterion.to(device)

'''----------------------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------------------------'''


def calculate_accuracy(y_pred, y):
    top_pred = (y_pred>0.5).float()
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

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
