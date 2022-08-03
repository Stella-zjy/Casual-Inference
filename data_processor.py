import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from scipy import stats
from matplotlib import style
import seaborn as sns
from matplotlib import pyplot as plt
import statsmodels.formula.api as smf
import graphviz as gr
from linearmodels.iv import IV2SLS

pd.set_option("display.max_columns", 5)
style.use("fivethirtyeight")

def get_dataset():
    data = pd.read_csv("../Casual-Inference/data/income_data/train.csv")
    data = data.dropna()
    data.describe()
    data = data.loc[:, ["race", "educational-num", "age", "income_>50K", "occupation", "relationship", "marital-status", "native-country", "gender"]]
    data.rename(columns = {'educational-num':'educational_num', "income_>50K": "income_bigger_than_50K", 'marital-status':'marital_status'}, inplace = True)
    data["race"] = data["race"].replace(to_replace ="Amer-Indian-Eskimo",
                     value ="Indian")
    data["race"] = data["race"].replace(to_replace ="Asian-Pac-Islander",
                     value ="Asian")
    occupationDict = {
        "Exec-managerial": 0,
        "Other-service": 1,
        "Transport-moving": 2,
        "Adm-clerical": 3,
        "Machine-op-inspct": 4,
        "Sales": 5,
        "Handlers-cleaners": 6,
        "Farming-fishing": 7,
        "Protective-serv": 8,
        "Prof-specialty": 9,
        "Craft-repair": 10,
        "Tech-support": 11,
        "Priv-house-serv": 12,
        "Armed-Forces": 13,
        "": -1
    }
    raceDict = {
        "White": 0,
        "Black": 1,
        "Asian": 2,
        "Indian": 3,
        "Other": 4,
        "": -1
    }
    genderDict = {
        "Male" : 0,
        "Female" : 1,
        "" : -1
    }
    def map_age(age):
        if age < 20 or age > 90:
            return -1
        if 20 <= age < 25:
            return 1
        if 25 <= age < 30:
            return 2
        if 30 <= age < 35:
            return 3
        if 35 <= age < 40:
            return 4
        if 40 <= age < 45:
            return 5
        if 45 <= age < 50:
            return 6
        if 50 <= age < 60:
            return 7
        if 60 <= age < 75:
            return 8
        if 75 <= age < 90:
            return 9

    data["occupation"] = data["occupation"].map(occupationDict)
    data["race"] = data["race"].map(raceDict)
    data["gender"] = data["gender"].map(genderDict)
    data["age_interval"] = data["age"].map(map_age)
    return data