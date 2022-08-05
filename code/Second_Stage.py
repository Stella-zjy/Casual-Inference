import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import time

from data_processor import solve_sample



class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # self.criterion = criterion
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
        return y_pred


    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def calculate_accuracy(self, y_pred, y):
        top_pred = (y_pred>0.5).float()
        correct = top_pred.eq(y.view_as(top_pred)).sum()
        acc = correct.float() / y.shape[0]
        return acc

  
    def fit(self, x, y, prob, EPOCHS=50, lr=0.005, batch_size = 64):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        num_sample = len(x)
        total_batch = num_sample // batch_size
        
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        prob = torch.tensor(prob, dtype=torch.float32)


        for epoch in range(EPOCHS):
            start_time = time.monotonic()

            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            
            epoch_loss = 0

            # Mini-batch Training
            for batch_num in range(total_batch):
                selected_idx = all_idx[batch_size*batch_num:(batch_num+1)*batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
                sub_prob = prob[selected_idx]

                optimizer.zero_grad()
                pred_y = self.forward(sub_x)
                
                xent_loss = 0
                for i in range(len(selected_idx)):
                    xent_loss += (sub_y[i] - sub_prob[i] * pred_y[i])**2
                          
                loss = xent_loss
                loss.backward()
                optimizer.step()
                epoch_loss += xent_loss

            pred_y_eval = model.forward(x)

            train_loss = epoch_loss.item() / num_sample
            train_acc = self.calculate_accuracy(pred_y_eval, y)

            end_time = time.monotonic()

            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)

            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')


    def predict(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        pred_y = self.forward(x)
        pred_y = (pred_y>0.5).float().detach()
        pred_y = torch.flatten(pred_y).numpy()
        return pred_y
        



# Stage 1 to Stage 2
xp, probability = solve_sample()
# Input value of the outcome network
x = xp[["workclass", "marital_status", "occupation", "relationship", "gender", 
        "native_country", "age", "education"]].to_numpy()
# Groudtruth value of the outcome network
y = xp[[">=50K"]].to_numpy()
# Probability
probability = probability.to_numpy()


# Initialize input & output dimensions
input_dim = x.shape[1]
output_dim = 1


###
model = MLP(input_dim, output_dim)
model.fit(x, y, probability)
# print(model.predict(x))