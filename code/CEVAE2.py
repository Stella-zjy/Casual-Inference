
import torch

import pyro
import pyro.distributions as dist
from pyro.contrib.cevae import CEVAE
import numpy as np
import pandas as pd
from data_processor import solve_sample, get_zxpy
from income2_data_processor import ak91Data, caEducationalData, IPUMSData


def main():

    sampleData = pd.read_csv("/Users/kunhanwu/Documents/GitHub/Casual-Inference/data/IPUMS_IncomeData/modified_IPUMS_IncomeData.csv")

    print(sampleData.columns)

    sampleData_train = sampleData.iloc[:int(len(sampleData) * 0.7)]
    sampleData_test = sampleData.iloc[int(len(sampleData) * 0.7):]

    x_train = torch.tensor(sampleData_train[["hours_per_week", 
    "marital_status", "occupation", "industry", 
    "gender", "age"]].to_numpy())
    x_train = x_train.to(torch.float)
    t_train = torch.tensor(sampleData_train["education"].to_numpy())
    t_train = t_train.to(torch.float)
    y_train = torch.tensor(sampleData_train["income"].to_numpy())
    y_train = y_train.to(torch.float)


    feature_dim = 6
    latent_dim = 20
    hidden_dim = 200
    num_layers = 3
    num_samples = 100
    cevae = CEVAE(
        feature_dim=feature_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_samples=num_samples,
    )

    num_epochs = 10
    batch_size = 64
    learning_rate = 1e-3
    learning_rate_decay = 0.01
    weight_decay = 1E-4
    cevae.fit(
        x_train,
        t_train,
        y_train,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        learning_rate_decay=learning_rate_decay,
        weight_decay=weight_decay,
    )

    x_test = torch.tensor(sampleData_test[["hours_per_week", 
    "marital_status", "occupation", "industry", 
    "gender", "age"]].to_numpy())
    x_test = x_test.to(torch.float)
    t_test = torch.tensor(sampleData_test["education"].to_numpy())
    t_test = t_test.to(torch.float)
    y_test = torch.tensor(sampleData_test["income"].to_numpy())
    y_test = y_test.to(torch.float)

    est_ite = cevae.ite(x_test)
    est_ate = est_ite.mean()
    print("estimated ATE = {:0.3g}".format(est_ate.item()))

if __name__ == "__main__":

    main()