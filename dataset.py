import torch
from torch.utils.data import Dataset

class imcomedataset(Dataset):
 
  def __init__(self,train_data,train_outcome,prob):

    self.x_train = torch.tensor(train_data, dtype=torch.float32)
    self.y_train = torch.tensor(train_outcome, dtype=torch.float32)
    self.prob = torch.from_numpy(prob.to_numpy())
 
  def __len__(self):
    return len(self.y_train)
   
  def __getitem__(self,idx):
    return self.x_train[idx],self.y_train[idx],self.prob[idx]