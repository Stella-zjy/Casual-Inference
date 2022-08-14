import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from income2_data_processor import IPUMSData
from sklearn.metrics import classification_report
# relative_path = "D:/Workspace/Casual-Inference"
relative_path = ".."
def two_stage_least_square():
    x, p = IPUMSData.get_stage1_input()
    clf = LinearRegression().fit(x, p)
    pred_p = clf.predict(x)
    for i in range(len(pred_p)):
        if pred_p[i] >= 0.5:
            pred_p[i] = 1.0
        else:
            pred_p[i] = 0.0
    print(classification_report(p, pred_p))
    x.insert(loc=len(x.columns), column="education", value=pred_p)
    del x["race"]
    y = pd.read_csv(relative_path + "/data/IPUMS_IncomeData/modified_IPUMS_IncomeData.csv")['income']
    clf = LinearRegression().fit(x, y)
    print(x.keys())
    pred_y = clf.predict(x)
    for i in range(len(pred_y)):
        if pred_y[i] >= 0.5:
            pred_y[i] = 1.0
        else:
            pred_y[i] = 0.0
    print(classification_report(y, pred_y))
    del x["education"]
    x.insert(loc=len(x.columns), column="education", value="1,0")
    x = x.drop('education', axis=1).join(
        x['education'].str.split(',', expand=True).stack().reset_index(level=1, drop=True).rename('education')) \
        .reset_index(level=0)
    xp = x[["age", "gender", "marital_status", "birth_place", "industry", "hours_per_week", "education"]]
    xp["education"] = pd.to_numeric(xp["education"])
    pred_y = clf.predict(xp)
    for i in range(len(pred_y)):
        if pred_y[i] >= 0.5:
            pred_y[i] = 1.0
        else:
            pred_y[i] = 0.0
    def calculate_ate(y_pred, y_fact):
        ate = 0
        size = len(y_fact)
        for i in range(size):
            if y_fact['education'][i] == 1:
                ite = y_fact['income'][i] - y_pred[2 * i + 1]
            else:
                ite = y_pred[2 * i] - y_fact['income'][i]
            ate += ite
        ate = ate / size
        return ate

    y_fact = pd.read_csv(relative_path + "/data/IPUMS_IncomeData/modified_IPUMS_IncomeData.csv")[['education', 'income']]
    print('ATE = '+ str(calculate_ate(pred_y, y_fact)))
if __name__ == '__main__':
    two_stage_least_square()
