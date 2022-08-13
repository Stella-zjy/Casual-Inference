import argparse
import imp
import logging


import torch

import pyro
import pyro.distributions as dist
from pyro.contrib.cevae import CEVAE
import numpy as np
import pandas as pd
from data_processor import solve_sample, get_zxpy

logging.getLogger("pyro").setLevel(logging.DEBUG)
logging.getLogger("pyro").handlers[0].setLevel(logging.DEBUG)


def main():
    # if args.cuda:
    #     torch.set_default_tensor_type("torch.cuda.FloatTensor")

    z1, z2, x, p, y = get_zxpy()
    x["education"] = p
    x["race"] = z2
    x[">=50K"] = y
    sampleData = x
    print(sampleData)

    sampleData_train = sampleData.iloc[:int(len(sampleData) * 0.7)]
    sampleData_test = sampleData.iloc[int(len(sampleData) * 0.7):]

    x_train = torch.tensor(sampleData_train[["workclass", 
    "marital_status", "occupation", "relationship", 
    "gender", "native_country"]].to_numpy())
    x_train = x_train.to(torch.float)
    t_train = torch.tensor(sampleData_train["education"].to_numpy())
    t_train = t_train.to(torch.float)
    y_train = torch.tensor(sampleData_train[">=50K"].to_numpy())
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

    x_test = torch.tensor(sampleData_test[["workclass", 
    "marital_status", "occupation", "relationship", 
    "gender", "native_country"]].to_numpy())
    x_test = x_test.to(torch.float)
    t_test = torch.tensor(sampleData_test["education"].to_numpy())
    t_test = t_test.to(torch.float)
    y_test = torch.tensor(sampleData_test[">=50K"].to_numpy())
    y_test = y_test.to(torch.float)


    naive_ate = y_test[t_test == 1].mean() - y_test[t_test == 0].mean()
    print("naive ATE = {:0.3g}".format(naive_ate))

    est_ite = cevae.ite(x_test)
    est_ate = est_ite.mean()
    print("estimated ATE = {:0.3g}".format(est_ate.item()))

if __name__ == "__main__":
    main()


