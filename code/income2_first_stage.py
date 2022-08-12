import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from income2_data_processor import ak91Data, caEducationalData, IPUMSData
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# this function will sample based on the probility and times give.
# the return is a list of samples
def repeat_sample(times=1, possiblity=0.5):
    temp = []

    def sample(num):
        u = np.random.rand()

        return 1 if u < num else 0

    for i in range(times):
        a = sample(possiblity)
        temp.append(a)

    return temp


# this function will build the new treatments
def build(x, probilities, times=1, save=False):
    treatments = [repeat_sample(times, i) for i in probilities]
    data = np.hstack((x.to_numpy(), treatments))
    new = pd.DataFrame(data, columns=list(x.columns)
                                     + ["p" + str(i) for i in range(0, times)])

    if save:
        new.to_csv("../data/income_data2/ca_sample_data.csv")
        print("Successful saved!")

    return new


# this function will do the first stage and return a new x.
# ml method will have 4 opinions
# save will auto save the new x with sample result to a csv file.
def solve_stage_one(ml_method="LR", save=False):
    #x, y = ak91Data().get_stage1_input()
    x, y = caEducationalData().get_stage1_input()
    #x, y = IPUMSData().get_stage1_input()
    d = {
        "LR": LogisticRegression(),
        "ADA": AdaBoostClassifier(),
        "FOREST": RandomForestClassifier(),
        "MLP": MLPClassifier(),
    }

    model = d[ml_method.upper()]
    clf = model.fit(x, y)
    pred = clf.predict(x)
    probilities = clf.predict_proba(x)[:, 1]
    print(classification_report(y, pred))
    new_x = build(x, probilities, 500, save)

    return new_x


if __name__ == '__main__':
    print(solve_stage_one("ADA", save=True))
