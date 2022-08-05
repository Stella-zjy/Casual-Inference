from tkinter.messagebox import RETRY
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
    data = pd.read_csv("../Casual-Inference/data/income_data/test.csv")
    data = data.dropna(axis=0)
    data.rename(columns={'educational-num': 'educational_num', "income_>50K": "income_bigger_than_50K",
                         'marital-status': 'marital_status', 'native-country': 'native_country'}, inplace=True)
    data["race"] = data["race"].replace(to_replace="Amer-Indian-Eskimo",
                                        value="Indian")
    data["race"] = data["race"].replace(to_replace="Asian-Pac-Islander",
                                        value="Asian")
    print(data.groupby('occupation').count())
    occupationDict = {
        "Exec-managerial": 0,
        "Other-service": 4,
        "Transport-moving": 5,
        "Adm-clerical": 6,
        "Machine-op-inspct": 5,
        "Sales": 3,
        "Handlers-cleaners": 5,
        "Farming-fishing": 5,
        "Protective-serv": 2,
        "Prof-specialty": 2,
        "Craft-repair": 1,
        "Tech-support": 5,
        "Priv-house-serv": 5,
        "Armed-Forces": 5
    }
    raceDict = {
        "White": 0,
        "Black": 1,
        "Asian": 2,
        "Indian": 2,
        "Other": 2
    }
    educationDict = {
        'Doctorate': 1,
        '12th': 0,
        'Bachelors': 1,
        '7th-8th': 0,
        'Some-college': 1,
        'HS-grad': 0,
        '9th': 0,
        '10th': 0,
        '11th': 0,
        'Masters': 1,
        'Preschool': 0,
        '5th-6th': 0,
        'Prof-school': 0,
        'Assoc-voc': 0,
        'Assoc-acdm': 0,
        '1st-4th': 0
    }
    genderDict = {
        "Male": 0,
        "Female": 1
    }
    maritalDict = {
        "Divorced": 2,
        "Never-married": 1,
        "Married-civ-spouse": 0,
        "Widowed": 2,
        "Separated": 2,
        "Married-spouse-absent": 2,
        "Married-AF-spouse": 2
    }
    workclassDict = {
        'Private': 0,
        'State-gov': 1,
        'Self-emp-not-inc': 2,
        'Federal-gov': 1,
        'Local-gov': 1,
        'Self-emp-inc': 1,
        'Without-pay': 1
    }

    def map_relationship(relationship):
        if relationship == "Husband":
            return 0
        if relationship == "Not-in-family":
            return 1
        else:
            return 2

    def map_country(native_country):
        if native_country == "United-States":
            return 0
        else:
            return 1

    data["occupation"] = data["occupation"].map(occupationDict)
    data["race"] = data["race"].map(raceDict)
    data["gender"] = data["gender"].map(genderDict)
    data["marital_status"] = data["marital_status"].map(maritalDict)
    data["native_country"] = data["native_country"].map(map_country)
    data["workclass"] = data["workclass"].map(workclassDict)
    data["education"] = data["education"].map(educationDict)
    data["relationship"] = data["relationship"].map(map_relationship)
    data.to_csv("modified_test.csv")
    return data

def solve_sample():
    data = pd.read_csv("../Casual-Inference/new X.csv")
    p = data.iloc[:, 8:].apply(lambda x:x.mean(), axis=1)
    data.insert(loc=len(data.columns), column='education', value="1,0")
    data = pd.concat([data, p], axis=1).rename(columns={0:'probability'})
    y = pd.read_csv("../Casual-Inference/data/income_data/modified_train.csv")["income_bigger_than_50K"]
    data = pd.concat([data, y], axis=1).rename(columns={"income_bigger_than_50K": ">=50K"})
    df_split_row = data.drop('education', axis=1).join(
        data['education'].str.split(',', expand=True).stack().reset_index(level=1, drop=True).rename('education'))\
        .reset_index(level=0)
    probability = df_split_row['probability']
    data = df_split_row.drop('probability', axis=1)
    for key in probability.keys():
        if key % 2 == 1:
            probability.iloc[key] = 1 - probability.iloc[key]
    xp = data[["workclass", "marital_status", "occupation", "relationship", "gender", "native_country", "age",
               "education", ">=50K"]]
    xp["education"] = pd.to_numeric(xp["education"])
    print(type(xp.iloc[0]["education"]))
    return xp, probability


def get_zxpy():
    data = pd.read_csv("../Casual-Inference/data/income_data/modified_train.csv")
    p = data["education"]
    x = data[[
        "workclass",  "marital_status", "occupation", "relationship", "gender", "hours-per-week", "native_country"]]
    z1 = data["age"]
    z2 = data["race"]
    y = data["income_bigger_than_50K"]
    return z1, z2, x, p, y

# get dataset ready for first stage ML, here we will use age as IV
def get_xp():
    data = pd.read_csv("../Casual-Inference/data/income_data/modified_train.csv")
    p = data["education"]
    x = data[[
        "workclass", "age", "marital_status", "occupation", "relationship", 
        "gender", "native_country"]]
    return x,p

if __name__ == '__main__':
    solve_sample()

